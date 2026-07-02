#ifndef INFERENCE_CPP_H
#define INFERENCE_CPP_H

#include <QString>
#include <QImage>
#include <QVector>
#include <QJsonArray>
#include <QJsonObject>
#include <onnxruntime_c_api.h>

// 单个检测结果
struct CppDetection {
    QString label;
    double  confidence;
    double  x, y, w, h;   // 原图像素坐标
};

// C++ ONNX Runtime 推理器
class CppInference {
public:
    CppInference();
    ~CppInference();

    // 加载 ONNX 模型
    bool loadModel(const QString &modelPath);

    // 是否已加载模型
    bool isLoaded() const { return m_loaded; }

    // 对单张图片执行推理
    // 返回 QJsonArray（与 Python 版输出格式兼容）
    QJsonArray inference(const QString &imagePath,
                         double confThresh = 0.25,
                         double iouThresh  = 0.45);

    // 获取最后一次推理的原始检测结果（供 showInferImage 使用）
    const QVector<CppDetection> &lastDetections() const { return m_detections; }

    // 错误信息
    QString lastError() const { return m_lastError; }

    // 输入尺寸
    int inputWidth()  const { return m_inputW; }
    int inputHeight() const { return m_inputH; }

    // 类别名
    QStringList classNames() const { return m_classNames; }

private:
    // 预处理：QImage → float CHW [0,1] normalized
    bool preprocess(const QImage &img, float *data);

    // 后处理：解析输出张量 + NMS
    QVector<CppDetection> postprocess(const float *output,
                                       int numBoxes, int numCls,
                                       int origW, int origH,
                                       double confThresh, double iouThresh);

    // NMS
    static QVector<int> nms(const QVector<CppDetection> &dets, double iouThresh);

    void clear();

    void *m_api      = nullptr;    // const OrtApi*（C++ 模式下的 ort API 入口）
    OrtSession *m_session = nullptr;
    OrtEnv    *m_env     = nullptr;
    OrtAllocator *m_memory = nullptr;

    QString m_lastError;
    bool    m_loaded = false;

    int m_inputW  = 640;
    int m_inputH  = 640;
    int m_numCls  = 1;

    // 输入/输出节点名
    char *m_inputName  = nullptr;
    char *m_outputName = nullptr;

    QStringList m_classNames;
    QVector<CppDetection> m_detections;
};

#endif // INFERENCE_CPP_H
