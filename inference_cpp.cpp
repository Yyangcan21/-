#include "inference_cpp.h"
#include "onnxruntime_c_api.h"

#include <QFile>
#include <QFileInfo>
#include <QDebug>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonDocument>

#include <algorithm>
#include <cmath>

// ============================================================
// 辅助：获取 ONNX Runtime API
// ============================================================

static const OrtApi *g_ort = nullptr;

static const OrtApi *ortApi() {
    if (!g_ort) {
        g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    }
    return g_ort;
}

// ============================================================
// 辅助：ONNX Runtime 状态码 → 字符串
// ============================================================

static QString ortErrorMsg(OrtStatus *status) {
    if (!status) return "No error";
    const char *msg = ortApi()->GetErrorMessage(status);
    QString s = QString::fromUtf8(msg);
    ortApi()->ReleaseStatus(status);
    return s;
}

// ============================================================
// 构造 / 析构
// ============================================================

CppInference::CppInference() {}

CppInference::~CppInference() {
    clear();
}

void CppInference::clear() {
    m_loaded = false;
    m_inputW = 640;
    m_inputH = 640;
    m_numCls = 1;
    m_detections.clear();
    m_classNames.clear();

    const OrtApi *api = ortApi();
    if (m_session) {
        api->ReleaseSession(m_session);
        m_session = nullptr;
    }
    if (m_inputName) {
        (void)api->AllocatorFree(m_memory, m_inputName);
        m_inputName = nullptr;
    }
    if (m_outputName) {
        (void)api->AllocatorFree(m_memory, m_outputName);
        m_outputName = nullptr;
    }
    if (m_env) {
        api->ReleaseEnv(m_env);
        m_env = nullptr;
    }
    m_memory = nullptr;
}

// ============================================================
// 加载 ONNX 模型
// ============================================================

bool CppInference::loadModel(const QString &modelPath) {
    clear();

    const OrtApi *api = ortApi();

    // 创建 Env
    OrtStatus *status = api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "CppInference", &m_env);
    if (status) {
        m_lastError = "CreateEnv failed: " + ortErrorMsg(status);
        return false;
    }

    // 创建 Session
    OrtSessionOptions *sessOpts = nullptr;
    (void)api->CreateSessionOptions(&sessOpts);
    (void)api->SetSessionGraphOptimizationLevel(sessOpts, ORT_ENABLE_ALL);

    // CPU provider is default, but explicitly enable it
    // OrtCUDAProviderOptions is not needed for now

    // ORT 在 Windows 上 ORTCHAR_T = wchar_t，要传宽字符
    const wchar_t *path = reinterpret_cast<const wchar_t *>(modelPath.utf16());
    status = api->CreateSession(m_env, path, sessOpts, &m_session);
    api->ReleaseSessionOptions(sessOpts);

    if (status) {
        m_lastError = "CreateSession failed: " + ortErrorMsg(status);
        api->ReleaseEnv(m_env);
        m_env = nullptr;
        return false;
    }

    // 获取输入输出信息
    (void)api->GetAllocatorWithDefaultOptions(&m_memory);

    size_t numInputs;
    status = api->SessionGetInputCount(m_session, &numInputs);
    if (status || numInputs < 1) {
        m_lastError = "No inputs found";
        return false;
    }

    // 获取第1个输入名称和形状
    status = api->SessionGetInputName(m_session, 0, m_memory, &m_inputName);
    if (status) {
        m_lastError = "GetInputName failed: " + ortErrorMsg(status);
        return false;
    }

    OrtTypeInfo *typeInfo = nullptr;
    (void)api->SessionGetInputTypeInfo(m_session, 0, &typeInfo);
    const OrtTensorTypeAndShapeInfo *tensorInfo = nullptr;
    (void)api->CastTypeInfoToTensorInfo(typeInfo, &tensorInfo);

    ONNXTensorElementDataType elemType;
    (void)api->GetTensorElementType(tensorInfo, &elemType);

    size_t numDims = 0;
    (void)api->GetDimensionsCount(tensorInfo, &numDims);
    int64_t dims[4] = {1, 3, 640, 640};
    if (numDims >= 2) {
        (void)api->GetDimensions(tensorInfo, dims, numDims);
    }

    if (numDims >= 4) {
        m_inputH = (int)dims[2];
        m_inputW = (int)dims[3];
    } else if (numDims == 3) {
        m_inputH = (int)dims[1];
        m_inputW = (int)dims[2];
    }

    api->ReleaseTensorTypeAndShapeInfo(const_cast<OrtTensorTypeAndShapeInfo *>(tensorInfo));
    api->ReleaseTypeInfo(typeInfo);

    // 获取输出名称
    size_t numOutputs;
    (void)api->SessionGetOutputCount(m_session, &numOutputs);
    if (numOutputs < 1) {
        m_lastError = "No outputs found";
        return false;
    }
    status = api->SessionGetOutputName(m_session, 0, m_memory, &m_outputName);
    if (status) {
        m_lastError = "GetOutputName failed: " + ortErrorMsg(status);
        return false;
    }

    // 推断类别数：从输出张量形状 [1, 4+nc, num_boxes]
    (void)api->SessionGetOutputTypeInfo(m_session, 0, &typeInfo);
    (void)api->CastTypeInfoToTensorInfo(typeInfo, &tensorInfo);
    (void)api->GetDimensionsCount(tensorInfo, &numDims);
    int64_t outDims[3] = {1, 5, 8400};
    if (numDims <= 3) {
        (void)api->GetDimensions(tensorInfo, outDims, numDims);
    }
    int outCols = 5;  // default: 4 box + 1 class
    if (numDims >= 2)
        outCols = (int)outDims[1];
    m_numCls = outCols - 4;
    if (m_numCls < 1) m_numCls = 1;

    api->ReleaseTensorTypeAndShapeInfo(const_cast<OrtTensorTypeAndShapeInfo *>(tensorInfo));
    api->ReleaseTypeInfo(typeInfo);

    // 尝试从模型元数据读取类别名
    OrtModelMetadata *meta = nullptr;
    status = api->SessionGetModelMetadata(m_session, &meta);
    if (!status && meta) {
        // 简单处理：为每个类别生成默认名称
        // 实际项目中可以从 meta custom_metadata_map 读取
        api->ReleaseModelMetadata(meta);
    }

    // 生成默认类别名（水坑检测通常 1 个类）
    if (m_numCls == 1) {
        m_classNames << "puddle";
    } else {
        for (int i = 0; i < m_numCls; ++i)
            m_classNames << QString("cls%1").arg(i);
    }

    m_loaded = true;
    qDebug() << "[CppInference] Model loaded:"
             << "input" << m_inputW << "x" << m_inputH
             << "classes" << m_numCls;
    return true;
}

// ============================================================
// 预处理：QImage → float CHW [0,1]
// ============================================================

bool CppInference::preprocess(const QImage &img, float *data) {
    if (img.isNull()) return false;

    // 缩放 + 转换为 RGB
    QImage scaled = img.scaled(m_inputW, m_inputH, Qt::KeepAspectRatioByExpanding,
                                Qt::SmoothTransformation);
    // 居中裁剪到精确尺寸
    int dx = (scaled.width()  - m_inputW) / 2;
    int dy = (scaled.height() - m_inputH) / 2;
    scaled = scaled.copy(dx, dy, m_inputW, m_inputH);

    // 强制转为 RGB888 格式
    QImage rgb = scaled.convertToFormat(QImage::Format_RGB888);
    if (rgb.isNull()) {
        // 回退：用 Format_RGB32 再转
        rgb = scaled.convertToFormat(QImage::Format_RGB32);
        if (rgb.isNull()) return false;
    }

    int w = rgb.width();
    int h = rgb.height();
    const uchar *bits = rgb.constBits();
    int bpl = rgb.bytesPerLine();

    // HWC → CHW, normalize to [0,1]
    for (int c = 0; c < 3; ++c) {
        float *channel = data + c * h * w;
        for (int y = 0; y < h; ++y) {
            const uchar *row = bits + y * bpl;
            for (int x = 0; x < w; ++x) {
                // RGB888: 3 bytes per pixel
                channel[y * w + x] = row[x * 3 + c] / 255.0f;
            }
        }
    }
    return true;
}

// ============================================================
// NMS
// ============================================================

QVector<int> CppInference::nms(const QVector<CppDetection> &dets, double iouThresh) {
    QVector<int> keep;
    int n = dets.size();
    if (n == 0) return keep;

    // 计算面积
    QVector<float> areas(n);
    for (int i = 0; i < n; ++i) {
        areas[i] = dets[i].w * dets[i].h;
    }

    // 按置信度降序索引
    QVector<int> indices(n);
    for (int i = 0; i < n; ++i) indices[i] = i;
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return dets[a].confidence > dets[b].confidence;
    });

    while (!indices.isEmpty()) {
        int i = indices.takeFirst();
        keep.append(i);

        float x1_i = dets[i].x;
        float y1_i = dets[i].y;
        float x2_i = x1_i + dets[i].w;
        float y2_i = y1_i + dets[i].h;

        QVector<int> remaining;
        for (int j : indices) {
            float x1_j = dets[j].x;
            float y1_j = dets[j].y;
            float x2_j = x1_j + dets[j].w;
            float y2_j = y1_j + dets[j].h;

            float interW = qMax(0.0f, qMin(x2_i, x2_j) - qMax(x1_i, x1_j));
            float interH = qMax(0.0f, qMin(y2_i, y2_j) - qMax(y1_i, y1_j));
            float inter  = interW * interH;
            float iou    = inter / (areas[i] + areas[j] - inter + 1e-6f);

            if (iou <= iouThresh)
                remaining.append(j);
        }
        indices = remaining;
    }
    return keep;
}

// ============================================================
// 后处理：解析输出 + NMS
// ============================================================

QVector<CppDetection> CppInference::postprocess(
    const float *output, int numBoxes, int numCls,
    int origW, int origH, double confThresh, double iouThresh)
{
    QVector<CppDetection> dets;

    int numCols = 4 + numCls;  // cx,cy,w,h + class_scores...

    float scaleX = (float)origW / m_inputW;
    float scaleY = (float)origH / m_inputH;

    for (int i = 0; i < numBoxes; ++i) {
        const float *row = output + i * numCols;

        // 解析类别置信度
        float maxScore = 0;
        int bestCls = 0;
        for (int c = 0; c < numCls; ++c) {
            float s = row[4 + c];
            if (s > maxScore) {
                maxScore = s;
                bestCls = c;
            }
        }
        if (maxScore < confThresh) continue;

        // YOLOv8 ONNX 输出: cx, cy, w, h (640像素坐标)
        float cx = row[0];
        float cy = row[1];
        float w  = row[2];
        float h  = row[3];

        // 转换到原图像素坐标
        float x = (cx - w / 2) * scaleX;
        float y = (cy - h / 2) * scaleY;
        float bw = w * scaleX;
        float bh = h * scaleY;

        // 裁剪到图像边界
        if (x < 0) { bw += x; x = 0; }
        if (y < 0) { bh += y; y = 0; }
        if (x + bw > origW) bw = origW - x;
        if (y + bh > origH) bh = origH - y;
        if (bw <= 0 || bh <= 0) continue;

        CppDetection d;
        d.label      = (bestCls < m_classNames.size()) ? m_classNames[bestCls] : QString("cls%1").arg(bestCls);
        d.confidence = maxScore;
        d.x = x; d.y = y; d.w = bw; d.h = bh;
        dets.append(d);
    }

    // NMS
    if (iouThresh > 0 && !dets.isEmpty()) {
        QVector<int> keep = nms(dets, iouThresh);
        QVector<CppDetection> filtered;
        for (int k : keep) filtered.append(dets[k]);
        dets = filtered;
    }

    return dets;
}

// ============================================================
// 主推理函数
// ============================================================

QJsonArray CppInference::inference(const QString &imagePath,
                                     double confThresh, double iouThresh)
{
    QJsonArray emptyResult;

    if (!m_loaded) {
        m_lastError = "Model not loaded";
        return emptyResult;
    }

    const OrtApi *api = ortApi();

    // ── 读取图片 ──
    QImage img(imagePath);
    if (img.isNull()) {
        m_lastError = "Cannot load image: " + imagePath;
        return emptyResult;
    }
    int origW = img.width();
    int origH = img.height();

    // ── 预处理 ──
    int imgSize = 3 * m_inputH * m_inputW;
    float *inputData = new float[imgSize];
    if (!preprocess(img, inputData)) {
        delete[] inputData;
        m_lastError = "Preprocess failed";
        return emptyResult;
    }

    // ── 创建输入张量 ──
    OrtMemoryInfo *memInfo = nullptr;
    (void)api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memInfo);

    int64_t inputShape[4] = {1, 3, m_inputH, m_inputW};
    OrtValue *inputTensor = nullptr;
    OrtStatus *status = api->CreateTensorWithDataAsOrtValue(
        memInfo, inputData, imgSize * sizeof(float),
        inputShape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &inputTensor);
    (void)api->ReleaseMemoryInfo(memInfo);

    if (status) {
        m_lastError = "CreateTensor failed: " + ortErrorMsg(status);
        delete[] inputData;
        return emptyResult;
    }

    // ── 推理 ──
    const char *inputNames[]  = {m_inputName};
    const char *outputNames[] = {m_outputName};

    OrtValue *outputTensor = nullptr;
    status = api->Run(m_session, nullptr,
                      inputNames, (const OrtValue *const *)&inputTensor, 1,
                      outputNames, 1,
                      &outputTensor);

    api->ReleaseValue(inputTensor);
    delete[] inputData;

    if (status) {
        m_lastError = "Run failed: " + ortErrorMsg(status);
        return emptyResult;
    }

    // ── 获取输出数据 ──
    float *outputData = nullptr;
    status = api->GetTensorMutableData(outputTensor, (void **)&outputData);
    if (status) {
        m_lastError = "GetTensorData failed: " + ortErrorMsg(status);
        api->ReleaseValue(outputTensor);
        return emptyResult;
    }

    // 获取输出形状
    OrtTensorTypeAndShapeInfo *outInfo = nullptr;
    (void)api->GetTensorTypeAndShape(outputTensor, &outInfo);
    size_t outNumDims;
    (void)api->GetDimensionsCount(outInfo, &outNumDims);
    int64_t outDims[3] = {1, 4 + m_numCls, 8400};
    (void)api->GetDimensions(outInfo, outDims, outNumDims);
    int numBoxes = 8400;
    int numCols  = 4 + m_numCls;
    if (outNumDims >= 2) {
        // [1, N, C] or [1, C, N] format
        if (outNumDims == 3) {
            // Determine which axis is boxes and which is classes
            int64_t d1 = outDims[1];
            int64_t d2 = outDims[2];
            if (d1 == 4 + m_numCls || d1 - 4 < qMin(m_numCls + 1, 10)) {
                numCols  = (int)d1;
                numBoxes = (int)d2;
            } else {
                numBoxes = (int)d1;
                numCols  = (int)d2;
            }
        } else if (outNumDims == 2) {
            numCols  = (int)outDims[0];  // might be wrong; handle carefully
            numBoxes = (int)outDims[1];
        }
    }

    // ── 后处理 ──
    m_detections = postprocess(outputData, numBoxes, m_numCls,
                                origW, origH, confThresh, iouThresh);

    (void)api->ReleaseTensorTypeAndShapeInfo(outInfo);
    (void)api->ReleaseValue(outputTensor);

    // ── 组装 JSON 结果 ──
    QJsonArray results;
    for (const auto &d : m_detections) {
        QJsonObject obj;
        obj["label"] = d.label;
        obj["conf"]  = round(d.confidence * 10000) / 10000.0;
        QJsonArray box;
        box.append(round(d.x * 10) / 10.0);
        box.append(round(d.y * 10) / 10.0);
        box.append(round(d.w * 10) / 10.0);
        box.append(round(d.h * 10) / 10.0);
        obj["box"] = box;
        results.append(obj);
    }

    m_lastError.clear();
    qDebug() << "[CppInference] Done, found" << results.size() << "detections";
    return results;
}
