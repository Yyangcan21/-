#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QMenuBar>
#include <QToolBar>
#include <QAction>
#include <QActionGroup>
#include <QInputDialog>
#include <QMessageBox>
#include <QFileInfo>
#include <QDir>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QHeaderView>
#include <QStringConverter>
#include <QPainter>
#include <QFontMetrics>
#include <QSqlRecord>
#include <QSqlField>
#include <QTableWidgetItem>
#include <QFile>
#include <QTextStream>
#include <QStandardPaths>
#include <QApplication>
#include <QTimer>

// ==========================================
// 构造 & 析构
// ==========================================

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , m_process(new QProcess(this))
    , m_totalEpochs(0)
    , m_currentEpoch(0)
    , m_annotView(nullptr)
    , m_labelModel(nullptr)
    , m_autoProcess(new QProcess(this))
    , m_quantProcess(new QProcess(this))
    , m_inferProcess(new QProcess(this))
    , m_db(new DatabaseManager(this))
{
    ui->setupUi(this);

    // =============================================
    // ① 菜单栏（文件/编辑/视图/训练/帮助）
    // =============================================
    {
        // ---- 文件 ----
        QMenu *menuFile = menuBar()->addMenu("文件(&F)");
        QAction *actOpenDataset = menuFile->addAction("📁 打开数据集(data.yaml)...");
        connect(actOpenDataset, &QAction::triggered, this, &MainWindow::on_btn_selectDataset_clicked);
        QAction *actOpenImageDir = menuFile->addAction("🖼 打开图片文件夹...");
        connect(actOpenImageDir, &QAction::triggered, this, &MainWindow::on_btn_annotOpenFolder_clicked);
        menuFile->addSeparator();
        QAction *actSaveAnnot = menuFile->addAction("💾 保存标注");
        actSaveAnnot->setShortcut(QKeySequence::Save);
        connect(actSaveAnnot, &QAction::triggered, this, &MainWindow::on_btn_annotSave_clicked);
        QAction *actSaveStat = menuFile->addAction("📊 保存标注统计到数据库");
        connect(actSaveStat, &QAction::triggered, this, &MainWindow::on_btn_dbSaveAnnot_clicked);
        menuFile->addSeparator();
        QAction *actExportCsv = menuFile->addAction("📤 导出 CSV...");
        connect(actExportCsv, &QAction::triggered, this, &MainWindow::on_btn_dbExport_clicked);
        menuFile->addSeparator();
        QAction *actQuit = menuFile->addAction("❌ 退出");
        actQuit->setShortcut(QKeySequence::Quit);
        connect(actQuit, &QAction::triggered, this, &QWidget::close);

        // ---- 编辑 ----
        QMenu *menuEdit = menuBar()->addMenu("编辑(&E)");
        QAction *actUndo = menuEdit->addAction("↩ 撤销");
        actUndo->setShortcut(QKeySequence::Undo);
        connect(actUndo, &QAction::triggered, ui->annotationView, &AnnotationView::undo);
        QAction *actRedo = menuEdit->addAction("↪ 重做");
        actRedo->setShortcut(QKeySequence::Redo);
        connect(actRedo, &QAction::triggered, ui->annotationView, &AnnotationView::redo);
        menuEdit->addSeparator();
        QAction *actAddLabel = menuEdit->addAction("➕ 添加标签...");
        connect(actAddLabel, &QAction::triggered, this, &MainWindow::on_btn_annotAddLabel_clicked);
        QAction *actDelLabel = menuEdit->addAction("🗑 删除当前标签");
        connect(actDelLabel, &QAction::triggered, this, &MainWindow::on_btn_annotDelLabel_clicked);

        // ---- 视图 ----
        QMenu *menuView = menuBar()->addMenu("视图(&V)");
        // 工具栏可隐藏
        QAction *actToggleToolbar = menuView->addAction("🔧 显示/隐藏标注工具栏");
        actToggleToolbar->setCheckable(true);
        actToggleToolbar->setChecked(true);
        connect(actToggleToolbar, &QAction::toggled,
                ui->toolBar_annotation, &QToolBar::setVisible);
        menuView->addSeparator();
        QAction *actZoomIn = menuView->addAction("🔍 放大");
        actZoomIn->setShortcut(QKeySequence::ZoomIn);
        connect(actZoomIn, &QAction::triggered, ui->annotationView, &AnnotationView::zoomIn);
        QAction *actZoomOut = menuView->addAction("🔍 缩小");
        actZoomOut->setShortcut(QKeySequence::ZoomOut);
        connect(actZoomOut, &QAction::triggered, ui->annotationView, &AnnotationView::zoomOut);
        QAction *actZoomFit = menuView->addAction("⬜ 适应窗口");
        actZoomFit->setShortcut(Qt::Key_0);
        connect(actZoomFit, &QAction::triggered, ui->annotationView, &AnnotationView::zoomFit);
        menuView->addSeparator();
        QAction *actGotoTrain = menuView->addAction("📈 切换到训练页");
        connect(actGotoTrain, &QAction::triggered, this, [this]{
            ui->tabWidget->setCurrentWidget(ui->tab_training);
        });
        QAction *actGotoAnnot = menuView->addAction("🏷 切换到标注页");
        connect(actGotoAnnot, &QAction::triggered, this, [this]{
            ui->tabWidget->setCurrentWidget(ui->tab_annotation);
        });
        QAction *actGotoQuant = menuView->addAction("⚡ 切换到量化页");
        connect(actGotoQuant, &QAction::triggered, this, [this]{
            ui->tabWidget->setCurrentWidget(ui->tab_quant);
        });
        QAction *actGotoDB = menuView->addAction("🗄 切换到数据库页");
        connect(actGotoDB, &QAction::triggered, this, [this]{
            for (int i = 0; i < ui->tabWidget->count(); ++i) {
                if (ui->tabWidget->tabText(i).contains("数据库")) {
                    ui->tabWidget->setCurrentIndex(i); break;
                }
            }
        });

        // ---- 训练 ----
        QMenu *menuTrain = menuBar()->addMenu("训练(&T)");
        QAction *actStartTrain = menuTrain->addAction("🚀 开始训练");
        connect(actStartTrain, &QAction::triggered, this, &MainWindow::on_btn_train_clicked);
        QAction *actStopTrain = menuTrain->addAction("⏹ 停止训练");
        connect(actStopTrain, &QAction::triggered, this, &MainWindow::on_btn_stop_clicked);
        menuTrain->addSeparator();
        QAction *actClearLog = menuTrain->addAction("🗑 清空训练日志");
        connect(actClearLog, &QAction::triggered, this, &MainWindow::on_btn_clearLog_clicked);
        QAction *actClearCurve = menuTrain->addAction("📈 清空训练曲线");
        connect(actClearCurve, &QAction::triggered, this, [this]{
            ui->lossCurveView->clearCurves();
            m_lossClsHistory.clear();
            m_lossDflHistory.clear();
            m_mapHistory.clear();
            m_precHistory.clear();
            m_recHistory.clear();
            m_lastEpochInCurve = 0;
            appendLog("📈 训练曲线已清空。");
        });

        // ---- 帮助 ----
        QMenu *menuHelp = menuBar()->addMenu("帮助(&H)");
        QAction *actAbout = menuHelp->addAction("ℹ️ 关于...");
        connect(actAbout, &QAction::triggered, this, [this]{
            QMessageBox::about(this, "关于",
                "<h2>水坑检测训练与标注工具 v1.0</h2>"
                "<p>Qt 6 + C++ + Python（YOLOv8 + ONNX Runtime）</p>"
                "<p>支持：图像标注 / 模型训练 / 模型量化 / 图像推理</p>"
                "<hr>"
                "<p><b>技术栈：</b></p>"
                "<ul>"
                "<li>前端：Qt 6 Widgets</li>"
                "<li>训练：ultralytics YOLOv8（QProcess 调 Python）</li>"
                "<li>量化：ONNX Runtime Dynamic Quantization</li>"
                "<li>C++ 推理：ONNX Runtime C++ API</li>"
                "<li>数据库：SQLite</li>"
                "</ul>");
        });
    }

    // ---- 进度条初始化 ----
    ui->progressBar->setValue(0);
    ui->progressBar->setRange(0, 100);

    // ---- 图像尺寸下拉默认选 640 ----
    ui->combo_imgsz->setCurrentText("640");

    // ---- 绑定训练进程信号 ----
    m_process->setProcessChannelMode(QProcess::MergedChannels);

    connect(m_process, &QProcess::readyReadStandardOutput,
            this, &MainWindow::onProcessReadyRead);

    connect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &MainWindow::onProcessFinished);

    connect(m_process, &QProcess::errorOccurred,
            this, &MainWindow::onProcessError);

    appendLog("═══════════════════════════════════════");
    appendLog("  水坑检测 — 模型训练工具  已就绪");
    appendLog("═══════════════════════════════════════");
    appendLog("👉 请先选择数据集文件和训练参数，然后点击「开始训练」");

    // =============================================
    // 标注页签初始化
    // =============================================
    // Qt Designer auto-connect: 手动触发 on_<obj>_clicked 等槽连接
    // Qt 6 uic 不会自动调用 connectSlotsByName
    QMetaObject::connectSlotsByName(this);

    // 获取 AnnotationView 指针（由 uic 自动创建）
    m_annotView = ui->annotationView;
    if (!m_annotView) {
        qFatal("annotationView not found in UI!");
    }

    // ---- 标注工具栏 ----
    QActionGroup *toolGroup = new QActionGroup(this);
    toolGroup->setExclusive(true);

    QAction *actSelect = new QAction("🖱 选择");
    QAction *actDraw   = new QAction("✏️ 画框");

    actSelect->setCheckable(true);
    actSelect->setChecked(true);
    actSelect->setShortcut(Qt::Key_V);
    toolGroup->addAction(actSelect);

    actDraw->setCheckable(true);
    actDraw->setShortcut(Qt::Key_B);
    toolGroup->addAction(actDraw);

    connect(actSelect, &QAction::triggered, this, [this, actSelect, actDraw]() {
        m_annotView->setTool(AnnotationView::TOOL_SELECT);
        actSelect->setChecked(true);
        actDraw->setChecked(false);
    });
    connect(actDraw, &QAction::triggered, this, [this, actSelect, actDraw]() {
        m_annotView->setTool(AnnotationView::TOOL_DRAW);
        actDraw->setChecked(true);
        actSelect->setChecked(false);
    });

    // 标注画布工具变化信号：同步工具栏按钮状态
    connect(m_annotView, &AnnotationView::toolChanged, this,
            [actSelect, actDraw](AnnotationView::Tool tool) {
                bool isSelect = (tool == AnnotationView::TOOL_SELECT);
                actSelect->setChecked(isSelect);
                actDraw->setChecked(!isSelect);
            });

    ui->toolBar_annotation->addAction(actSelect);
    ui->toolBar_annotation->addAction(actDraw);

    ui->toolBar_annotation->addSeparator();

    QAction *actZoomIn = new QAction("🔍 放大", this);
    actZoomIn->setShortcut(Qt::Key_Equal);
    connect(actZoomIn, &QAction::triggered, m_annotView, &AnnotationView::zoomIn);
    ui->toolBar_annotation->addAction(actZoomIn);

    QAction *actZoomOut = new QAction("🔍 缩小", this);
    actZoomOut->setShortcut(Qt::Key_Minus);
    connect(actZoomOut, &QAction::triggered, m_annotView, &AnnotationView::zoomOut);
    ui->toolBar_annotation->addAction(actZoomOut);

    QAction *actZoomFit = new QAction("⬜ 适应窗口", this);
    actZoomFit->setShortcut(Qt::Key_0);
    connect(actZoomFit, &QAction::triggered, m_annotView, &AnnotationView::zoomFit);
    ui->toolBar_annotation->addAction(actZoomFit);

    ui->toolBar_annotation->addSeparator();

    QAction *actRotateCW = new QAction("↻ 顺时针旋转", this);
    actRotateCW->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_R));
    connect(actRotateCW, &QAction::triggered, m_annotView, &AnnotationView::rotateCW);
    ui->toolBar_annotation->addAction(actRotateCW);

    QAction *actRotateCCW = new QAction("↺ 逆时针旋转", this);
    connect(actRotateCCW, &QAction::triggered, m_annotView, &AnnotationView::rotateCCW);
    ui->toolBar_annotation->addAction(actRotateCCW);

    QAction *actRotateReset = new QAction("↕ 重置旋转", this);
    connect(actRotateReset, &QAction::triggered, m_annotView, &AnnotationView::rotateReset);
    ui->toolBar_annotation->addAction(actRotateReset);

    ui->toolBar_annotation->addSeparator();

    QAction *actUndo = new QAction("↩ 撤销", this);
    actUndo->setShortcut(QKeySequence::Undo);
    connect(actUndo, &QAction::triggered, m_annotView, &AnnotationView::undo);
    ui->toolBar_annotation->addAction(actUndo);

    QAction *actRedo = new QAction("↪ 重做", this);
    actRedo->setShortcut(QKeySequence::Redo);
    connect(actRedo, &QAction::triggered, m_annotView, &AnnotationView::redo);
    ui->toolBar_annotation->addAction(actRedo);

    ui->toolBar_annotation->addSeparator();

    QAction *actDel = new QAction("🗑 删除选中框", this);
    actDel->setShortcut(Qt::Key_Delete);
    connect(actDel, &QAction::triggered, this, [this]() {
        if (m_annotView->getBoxes().isEmpty()) return;
        // 删除所有选中的框（这里简化：删除所有）
        m_annotView->clearBoxes();
    });
    ui->toolBar_annotation->addAction(actDel);

    // ---- 标签表格 ----
    m_labelModel = new QStandardItemModel(0, 3, this);
    m_labelModel->setHeaderData(0, Qt::Horizontal, "标签名");
    m_labelModel->setHeaderData(1, Qt::Horizontal, "颜色");
    m_labelModel->setHeaderData(2, Qt::Horizontal, "显示");
    ui->tableView_labels->setModel(m_labelModel);
    ui->tableView_labels->setColumnWidth(0, 80);
    ui->tableView_labels->setColumnWidth(1, 30);
    ui->tableView_labels->setColumnWidth(2, 40);
    ui->tableView_labels->horizontalHeader()->setStretchLastSection(true);
    ui->tableView_labels->setSelectionBehavior(QAbstractItemView::SelectRows);

    // 连接标注画布的框变化信号
    connect(m_annotView, &AnnotationView::boxesChanged,
            this, &MainWindow::on_annotBoxesChanged);

    // ===== 选中框变化 → 刷新右侧"标注信息列表" =====
    connect(m_annotView, &AnnotationView::selectedBoxChanged,
            this, [this](int boxId, const QString &label,
                         double nx, double ny, double nw, double nh,
                         int px, int py, int pw, int ph) {
                Q_UNUSED(label); Q_UNUSED(nx); Q_UNUSED(ny);
                Q_UNUSED(nw); Q_UNUSED(nh);
                Q_UNUSED(px);  Q_UNUSED(py);  Q_UNUSED(pw); Q_UNUSED(ph);
                if (!ui || !ui->listWidget_annotInfo) return;
                if (boxId < 0) {
                    // 没选中时显示提示
                    ui->listWidget_annotInfo->clear();
                    ui->listWidget_annotInfo->addItem("（无选中：请点击画布上的某个框）");
                    return;
                }
                // 有选中 → 整列表刷新
                updateAnnotInfoList();
            });

    // ===== 标签表点击 → 同步"当前标签" + 尝试改已选中框的类别 =====
    connect(ui->tableView_labels, &QTableView::clicked,
            this, &MainWindow::on_tableView_labels_clicked);

    // ===== "改选中框类别" 按钮 =====
    connect(ui->btn_annotChangeLabel, &QPushButton::clicked,
            this, &MainWindow::on_btn_annotChangeLabel_clicked);

    // 添加默认标签
    annotAddDefaultLabels();

    // ---- 自动标注进程 ----
    connect(m_autoProcess, &QProcess::readyReadStandardOutput,
            this, &MainWindow::on_annotAutoReadyRead);
    connect(m_autoProcess, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &MainWindow::on_annotAutoFinished);

    // =============================================
    // 量化进程初始化
    // =============================================
    m_quantProcess->setProcessChannelMode(QProcess::MergedChannels);
    connect(m_quantProcess, &QProcess::readyReadStandardOutput,
            this, &MainWindow::onQuantReadyRead);
    connect(m_quantProcess, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &MainWindow::onQuantFinished);
    connect(m_quantProcess, &QProcess::errorOccurred,
            this, &MainWindow::onQuantError);

    // 量化进度条默认设为 busy（无限循环）
    ui->progressBar_quant->setRange(0, 0);  // busy indicator
    ui->progressBar_quant->setValue(0);

    // =============================================
    // 推理进程初始化
    // =============================================
    m_inferProcess->setProcessChannelMode(QProcess::MergedChannels);
    connect(m_inferProcess, &QProcess::readyReadStandardOutput,
            this, &MainWindow::onInferReadyRead);
    connect(m_inferProcess, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &MainWindow::onInferFinished);

    // =============================================
    // 数据库初始化
    // =============================================
    dbInit();
}

MainWindow::~MainWindow()
{
    // 若进程仍在运行，强制终止
    if (m_process->state() != QProcess::NotRunning) {
        m_process->kill();
        m_process->waitForFinished(3000);
    }
    if (m_quantProcess->state() != QProcess::NotRunning) {
        m_quantProcess->kill();
        m_quantProcess->waitForFinished(3000);
    }
    if (m_inferProcess->state() != QProcess::NotRunning) {
        m_inferProcess->kill();
        m_inferProcess->waitForFinished(3000);
    }
    delete ui;
}

// ==========================================
// 注解脚注（原图表自适应重绘已删除）
// ==========================================


// ==========================================
// 兼容旧 pushButton（防止编译找不到槽）
// ==========================================
void MainWindow::on_pushButton_clicked()
{
    // 保留空实现即可
}

// ==========================================
// 选择数据集文件
// ==========================================
void MainWindow::on_btn_selectDataset_clicked()
{
    QString defaultDir = "D:/YC/software/QT/PuddleDetection/dataset";
    QString filePath = QFileDialog::getOpenFileName(
        this,
        "选择 YOLO 数据集配置文件 (data.yaml)",
        defaultDir,
        "YAML Files (*.yaml *.yml);;All Files (*)"
    );

    if (!filePath.isEmpty()) {
        ui->lineEdit_dataset->setText(filePath);
        appendLog("✅ 数据集已选择：" + filePath);
        ui->statusbar->showMessage("数据集：" + filePath, 5000);
    }
}

// ==========================================
// 选择 Python 解释器
// ==========================================
void MainWindow::on_btn_selectPython_clicked()
{
    QString defaultDir = "D:/Python";
    QString filePath = QFileDialog::getOpenFileName(
        this,
        "选择 Python 解释器",
        defaultDir,
        "Executable (*.exe);;All Files (*)"
    );

    if (!filePath.isEmpty()) {
        ui->lineEdit_python->setText(filePath);
        appendLog("🐍 Python 路径已更新：" + filePath);
    }
}

// ==========================================
// 开始训练
// ==========================================
void MainWindow::on_btn_train_clicked()
{
    // ---- 参数校验 ----
    QString dataset = ui->lineEdit_dataset->text().trimmed();
    if (dataset.isEmpty()) {
        appendLog("❌ 错误：请先选择数据集配置文件（data.yaml）！");
        ui->statusbar->showMessage("请先选择数据集！", 3000);
        return;
    }
    if (!dataset.endsWith(".yaml", Qt::CaseInsensitive) &&
        !dataset.endsWith(".yml",  Qt::CaseInsensitive)) {
        appendLog("❌ 错误：所选文件不是 .yaml 格式，请重新选择！");
        return;
    }

    QString pythonPath = ui->lineEdit_python->text().trimmed();
    if (pythonPath.isEmpty()) {
        appendLog("❌ 错误：请填写 Python 解释器路径！");
        return;
    }

    // ---- 读取训练参数 ----
    m_totalEpochs   = ui->spin_epoch->value();
    int    batch    = ui->spin_batch->value();
    double lr       = ui->spin_lr->value();
    QString imgsz   = ui->combo_imgsz->currentText();
    int    workers  = ui->spin_workers->value();

    // 从下拉框中提取设备关键字
    // 当前索引 0=cpu, 1=GPU0, 2=GPU1
    int devIdx = ui->combo_device->currentIndex();
    QString device;
    if (devIdx == 0)      device = "cpu";
    else if (devIdx == 1) device = "0";   // GPU 0
    else                  device = "1";  // GPU 1

    // 从下拉框中提取模型文件名（取括号前部分）
    QString modelFull = ui->combo_model->currentText();
    QString modelName = modelFull.split("（").first().trimmed(); // e.g. "yolov8n.pt"

    // train.py 的绝对路径
    QString trainScript = "D:/YC/software/QT/PuddleDetection/train.py";
    m_trainScriptDir    = QFileInfo(trainScript).absolutePath();

    // ---- 组装参数列表 ----
    QStringList args;
    args << trainScript
         << dataset
         << QString::number(m_totalEpochs)
         << QString::number(batch)
         << QString::number(lr, 'f', 5)
         << imgsz
         << device
         << modelName
         << QString::number(workers);

    // ---- 重置进度与曲线历史 ----
    m_currentEpoch = 0;
    m_lossClsHistory.clear();
    m_lossDflHistory.clear();
    m_mapHistory.clear();
    m_precHistory.clear();
    m_recHistory.clear();
    m_lastEpochInCurve = 0;
    m_lastMap50     = -1.0;
    m_lastPrecision = -1.0;
    m_lastRecall    = -1.0;
    ui->lossCurveView->clearCurves();
    ui->progressBar->setRange(0, m_totalEpochs);
    ui->progressBar->setValue(0);

    // ---- 切换界面状态 ----
    setTrainingState(true);

    // ---- 打印启动信息 ----
    appendLog("");
    appendLog("════════════════ 训练启动 ════════════════");
    appendLog(QString("📅 时间   : %1").arg(QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss")));
    appendLog(QString("📁 数据集 : %1").arg(dataset));
    appendLog(QString("🤖 模型   : %1").arg(modelName));
    appendLog(QString("⚙️  轮数   : %1  批大小: %2  学习率: %3").arg(m_totalEpochs).arg(batch).arg(lr));
    appendLog(QString("📐 图像尺寸: %1  设备: %2  Workers: %3").arg(imgsz).arg(device).arg(workers));
    appendLog("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // ---- 启动进程 ----
    m_process->setWorkingDirectory(m_trainScriptDir);
    m_process->start(pythonPath, args);

    if (!m_process->waitForStarted(5000)) {
        appendLog("❌ 进程启动失败！请检查 Python 路径是否正确。");
        appendLog("   路径：" + pythonPath);
        setTrainingState(false);
        return;
    }
    ui->statusbar->showMessage("训练进行中...");
}

// ==========================================
// 停止训练
// ==========================================
void MainWindow::on_btn_stop_clicked()
{
    if (m_process->state() != QProcess::NotRunning) {
        m_process->terminate(); // 先发 SIGTERM
        if (!m_process->waitForFinished(3000)) {
            m_process->kill(); // 超时强杀
        }
        appendLog("");
        appendLog("⏹ 训练已被用户手动停止。");
    }
    setTrainingState(false);
}

// ==========================================
// 清空日志
// ==========================================
void MainWindow::on_btn_clearLog_clicked()
{
    ui->text_log->clear();
}

// ==========================================
// 进程实时输出
// ==========================================
void MainWindow::onProcessReadyRead()
{
    QByteArray raw = m_process->readAllStandardOutput();

    // Python 端设置了 UTF-8，Windows 也可能输出 GBK，优先 UTF-8
    QString text = QString::fromUtf8(raw);

    // 去除 ANSI 颜色转义码（例如 \x1b[34m）
    static const QRegularExpression ansiRe("\\x1b\\[[^a-zA-Z]*[a-zA-Z]");
    text.remove(ansiRe);

    // 去除多余的回车符（Windows CRLF 处理）
    text.replace("\r\n", "\n");
    text.replace("\r", "\n");

    // 按行分割，逐行追加（避免末尾空行刷屏）
    QStringList lines = text.split('\n');
    bool epochAdvanced = false;
    for (const QString &line : lines) {
        QString trimmed = line.trimmed();
        if (!trimmed.isEmpty()) {
            appendLog(trimmed);
            int prevEpoch = m_currentEpoch;
            tryParseEpoch(trimmed);
            tryParseMetrics(trimmed);
            if (m_currentEpoch != prevEpoch) epochAdvanced = true;
        }
    }
    // 每次 epoch 推进时刷新一次曲线（避免每行 log 都刷新 6 条曲线）
    if (epochAdvanced) {
        refreshLossCurve();
    }
}

// ==========================================
// 进程结束
// ==========================================
void MainWindow::onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    appendLog("");
    if (exitStatus == QProcess::NormalExit && exitCode == 0) {
        appendLog("════════════════ 训练完成 ════════════════");
        appendLog("🎉 训练已成功完成！");
        appendLog(QString("📅 结束时间: %1").arg(
            QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss")));
        appendLog("💾 结果保存于 runs/train/exp 目录下");
        appendLog("═══════════════════════════════════════");

        // 进度条拉满
        ui->progressBar->setValue(m_totalEpochs);
        ui->label_status->setText("训练完成 ✅");
        ui->label_status->setStyleSheet("color: #4CAF50; font-weight: bold;");
        ui->statusbar->showMessage("训练完成！", 0);

        // 输出解析到的性能指标
        if (m_lastMap50 > 0) {
            appendLog(QString("📊 性能指标: mAP50=%.3f  Precision=%.3f  Recall=%.3f")
                      .arg(m_lastMap50).arg(m_lastPrecision).arg(m_lastRecall));
        }
        // 最后刷新一次曲线（带上最终 mAP/P/R）
        refreshLossCurve();

        // ---- 写入数据库 ----
        if (m_db->isOpen()) {
            QString dataset   = ui->lineEdit_dataset->text().trimmed();
            QString modelFull = ui->combo_model->currentText();
            QString modelName = modelFull.split("（").first().trimmed();
            int     batch     = ui->spin_batch->value();
            double  lr        = ui->spin_lr->value();
            int     imgsz     = ui->combo_imgsz->currentText().toInt();
            int devIdx = ui->combo_device->currentIndex();
            QString device = (devIdx == 0) ? "cpu" : (devIdx == 1 ? "0" : "1");
            QString outputDir = m_trainScriptDir + "/runs/train";
            QString bestPt    = findBestPt(outputDir);
            if (!bestPt.isEmpty()) {
                appendLog("📦 模型文件: " + bestPt);
            }
            qint64 rid = m_db->insertTrainRecord(dataset, modelName,
                                                 m_totalEpochs, batch, lr,
                                                 imgsz, device,
                                                 "completed",
                                                 m_lastMap50,
                                                 m_lastPrecision,
                                                 m_lastRecall,
                                                 outputDir,
                                                 bestPt);
            if (rid > 0) {
                appendLog(QString("🗄  训练记录已保存到数据库（ID=%1）").arg(rid));
                dbRefreshStats();
            }
        }

        // 重置指标，为下次训练做准备
        m_lastMap50     = -1.0;
        m_lastPrecision = -1.0;
        m_lastRecall    = -1.0;
    } else {
        appendLog("════════════════ 训练异常 ════════════════");
        appendLog(QString("❌ 进程以退出码 %1 结束（%2）")
                  .arg(exitCode)
                  .arg(exitStatus == QProcess::CrashExit ? "进程崩溃" : "异常退出"));
        appendLog("请检查上方日志中的错误信息。");
        appendLog("═══════════════════════════════════════");

        ui->label_status->setText("训练失败 ❌");
        ui->label_status->setStyleSheet("color: #f44336; font-weight: bold;");
        ui->statusbar->showMessage("训练异常结束，请查看日志。", 0);

        // ---- 写入失败记录 ----
        if (m_db->isOpen()) {
            QString dataset   = ui->lineEdit_dataset->text().trimmed();
            QString modelFull = ui->combo_model->currentText();
            QString modelName = modelFull.split("（").first().trimmed();
            m_db->insertTrainRecord(dataset, modelName,
                                    m_totalEpochs,
                                    ui->spin_batch->value(),
                                    ui->spin_lr->value(),
                                    ui->combo_imgsz->currentText().toInt(),
                                    (ui->combo_device->currentIndex() == 0) ? "cpu" : "gpu",
                                    exitStatus == QProcess::CrashExit ? "crashed" : "failed",
                                    -1.0, -1.0, -1.0, QString(), QString());
        }
    }

    setTrainingState(false);
}

// ==========================================
// 进程错误（启动失败等）
// ==========================================
void MainWindow::onProcessError(QProcess::ProcessError error)
{
    QString errMsg;
    switch (error) {
        case QProcess::FailedToStart: errMsg = "进程启动失败（Python 路径错误或权限不足）"; break;
        case QProcess::Crashed:       errMsg = "进程意外崩溃";                              break;
        case QProcess::Timedout:      errMsg = "进程超时";                                  break;
        case QProcess::WriteError:    errMsg = "写入进程失败";                              break;
        case QProcess::ReadError:     errMsg = "读取进程输出失败";                          break;
        default:                      errMsg = "未知错误";                                  break;
    }
    appendLog("❌ 进程错误：" + errMsg);
    setTrainingState(false);
}

// ==========================================
// 工具：追加日志
// ==========================================
void MainWindow::appendLog(const QString &text)
{
    ui->text_log->appendPlainText(text);
    // 自动滚动到底部
    QScrollBar *sb = ui->text_log->verticalScrollBar();
    sb->setValue(sb->maximum());
}

// ==========================================
// 工具：切换界面训练状态
// ==========================================
void MainWindow::setTrainingState(bool training)
{
    ui->btn_train->setEnabled(!training);
    ui->btn_stop->setEnabled(training);
    ui->btn_selectDataset->setEnabled(!training);
    ui->btn_selectPython->setEnabled(!training);
    ui->spin_epoch->setEnabled(!training);
    ui->spin_batch->setEnabled(!training);
    ui->spin_lr->setEnabled(!training);
    ui->spin_workers->setEnabled(!training);
    ui->combo_model->setEnabled(!training);
    ui->combo_device->setEnabled(!training);
    ui->combo_imgsz->setEnabled(!training);

    if (training) {
        ui->label_status->setText("训练中... ⏳");
        ui->label_status->setStyleSheet("color: #FF9800; font-weight: bold;");
    } else {
        // 仅在不是"完成/失败"时才重置为"等待"
        if (ui->label_status->text().startsWith("训练中")) {
            ui->label_status->setText("等待开始");
            ui->label_status->setStyleSheet("color: #888; font-weight: bold;");
        }
    }
}

// ==========================================
// 工具：从日志行解析 epoch 进度
// ==========================================
void MainWindow::tryParseEpoch(const QString &line)
{
    // YOLOv8 的进度行格式类似：
    //   "  1/50    ...  " 或 "Epoch 1/50"
    // 匹配 "当前epoch/总epoch" 形式
    static const QRegularExpression reSlash(R"((\d+)/(\d+))");
    QRegularExpressionMatch m = reSlash.match(line);
    if (m.hasMatch()) {
        int cur   = m.captured(1).toInt();
        int total = m.captured(2).toInt();
        // 简单过滤：total 要和用户设置的轮数接近（允许±1），cur <= total
        if (total > 0 && cur > 0 && cur <= total &&
            (total == m_totalEpochs || qAbs(total - m_totalEpochs) <= 1)) {
            m_currentEpoch = cur;
            ui->progressBar->setRange(0, total);
            ui->progressBar->setValue(cur);
            ui->label_status->setText(QString("第 %1 / %2 轮").arg(cur).arg(total));
        }
    }

    // ---- 解析 box_loss / cls_loss / dfl_loss ----
    // YOLOv8 每行格式：
    //   "  1/50      2.34G      1.234      0.567      0.890         45        640: 100%|██████████| ..."
    // 用更宽松的正则：找 "数字/数字" 后面 3 个 float
    static const QRegularExpression reLoss(
        R"(^\s*(\d+)\s*/\s*(\d+)[^|\n]*?(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+))"
    );
    QRegularExpressionMatch mLoss = reLoss.match(line);
    if (mLoss.hasMatch()) {
        int cur   = mLoss.captured(1).toInt();
        // 注意：YOLOv8 有些版本输出 5 个数字（box cls dfl + Instances+Size），
        // 严格只取前 3 个
        // box_loss 已废弃不再绘制，仅占位匹配正则
        double clsLoss = mLoss.captured(4).toDouble();
        double dflLoss = mLoss.captured(5).toDouble();
        if (cur > 0 && cur <= m_totalEpochs + 1) {
            m_lossClsHistory.append(QPointF(cur, clsLoss));
            m_lossDflHistory.append(QPointF(cur, dflLoss));
        }
    }
}

// ==========================================
// 工具：从日志行解析 YOLO 验证指标 (P/R/mAP50)
// ==========================================
void MainWindow::tryParseMetrics(const QString &line)
{
    // YOLOv8 验证结果行格式：
    // "all        100        500      0.612      0.579      0.548      0.321"
    static const QRegularExpression reMetrics(
        R"(all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+)"
    );
    QRegularExpressionMatch m = reMetrics.match(line);
    if (m.hasMatch()) {
        double p   = m.captured(1).toDouble();  // Precision (Box(P))
        double r   = m.captured(2).toDouble();  // Recall (R)
        double m50 = m.captured(3).toDouble();  // mAP50

        if (p >= 0 && p <= 1 && r >= 0 && r <= 1 && m50 >= 0 && m50 <= 1) {
            m_lastPrecision = p;
            m_lastRecall    = r;
            m_lastMap50     = m50;
            // 同时写入历史数组（用当前 epoch 作为 x 轴）
            if (m_currentEpoch > 0) {
                m_precHistory.append(QPointF(m_currentEpoch, p));
                m_recHistory.append(QPointF(m_currentEpoch, r));
                m_mapHistory.append(QPointF(m_currentEpoch, m50));
            }
        }
    }
}

// ==========================================
// 把所有曲线数据"提交"到 LossCurveView（避免每行 log 都刷新 6 条曲线）
// ==========================================
void MainWindow::refreshLossCurve()
{
    if (!ui->lossCurveView) return;
    ui->lossCurveView->setMaxEpoch(m_totalEpochs);
    ui->lossCurveView->clearCurves();
    for (int i = 0; i < m_lossClsHistory.size(); ++i) {
        int ep = int(m_lossClsHistory[i].x());
        double cl = m_lossClsHistory[i].y();
        double df = (i < m_lossDflHistory.size()) ? m_lossDflHistory[i].y() : -1.0;
        // mAP/P/R 按索引从历史数组取，取不到才 fallback 到 -1
        double m50  = (i < m_mapHistory.size())  ? m_mapHistory[i].y()  : -1.0;
        double prec = (i < m_precHistory.size()) ? m_precHistory[i].y() : -1.0;
        double rec  = (i < m_recHistory.size())  ? m_recHistory[i].y()  : -1.0;
        ui->lossCurveView->addDataPoint(ep, cl, df, m50, prec, rec);
    }
    m_lastEpochInCurve = m_currentEpoch;
}

// ==========================================
// 工具：在 outputDir 下找最新的 best.pt
// ==========================================
QString MainWindow::findBestPt(const QString &outputDir)
{
    QDir dir(outputDir);
    if (!dir.exists()) return QString();

    // 扫描 runs/train/ 下所有 expN 子目录，按时间倒序，找 weights/best.pt
    QFileInfoList exps = dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Time);
    for (const QFileInfo &exp : exps) {
        QString bestPt = exp.absoluteFilePath() + "/weights/best.pt";
        if (QFile::exists(bestPt))
            return QDir::toNativeSeparators(bestPt);
    }
    return QString();
}

// ============================================================
// 标注工具函数
// ============================================================

static QColor nextAnnotColor(int idx) {
    const QList<QColor> pool = {
        QColor("#FF6B6B"), QColor("#4ECDC4"), QColor("#45B7D1"),
        QColor("#96CEB4"), QColor("#FFEAA7"), QColor("#DDA0DD"),
        QColor("#98D8C8"), QColor("#F7DC6F"), QColor("#BB8FCE"),
        QColor("#85C1E9"), QColor("#F8B500"), QColor("#00CED1"),
    };
    return pool[idx % pool.size()];
}

QString MainWindow::annotImagePathToLabelPath(const QString &imgPath) const
{
    QFileInfo fi(imgPath);
    return fi.absolutePath() + "/" + fi.baseName() + ".txt";
}

void MainWindow::annotAddDefaultLabels()
{
    m_labelModel->insertRow(0, QList<QStandardItem *>()
        << new QStandardItem("puddle")
        << new QStandardItem("■")
        << new QStandardItem("✓"));
    annotSyncLabels();
}

void MainWindow::annotSyncLabels()
{
    QVector<LabelInfo> labels;
    for (int i = 0; i < m_labelModel->rowCount(); ++i) {
        LabelInfo li;
        li.name = m_labelModel->item(i, 0)->text();
        li.color = nextAnnotColor(i);
        li.visible = (m_labelModel->item(i, 2)->text() == "✓");
        labels.append(li);
    }
    m_annotView->setLabels(labels);
}

void MainWindow::annotLoadImage(const QString &path)
{
    if (!QFile::exists(path)) return;

    QImage img(path);
    if (img.isNull()) {
        QMessageBox::warning(this, "错误", "无法加载图片：" + path);
        return;
    }

    m_annotView->setImage(img);

    // 加载已有标注
    QString labelPath = annotImagePathToLabelPath(path);
    QVector<BoundingBox> boxes;

    if (QFile::exists(labelPath)) {
        QFile f(labelPath);
        if (f.open(QIODevice::ReadOnly | QIODevice::Text)) {
            QTextStream ts(&f);
            ts.setEncoding(QStringConverter::Utf8);
            int id = 0;
            while (!ts.atEnd()) {
                QString line = ts.readLine().trimmed();
                if (line.isEmpty()) continue;
                QStringList parts = line.split(' ', Qt::SkipEmptyParts);
                if (parts.size() < 5) continue;
                // YOLO 格式: class_id cx cy w h
                int classId = parts[0].toInt();
                bool ok1, ok2, ok3, ok4;
                double cx = parts[1].toDouble(&ok1);
                double cy = parts[2].toDouble(&ok2);
                double w  = parts[3].toDouble(&ok3);
                double h  = parts[4].toDouble(&ok4);
                if (!ok1 || !ok2 || !ok3 || !ok4) continue;

                BoundingBox box;
                box.id = id++;
                box.rect = QRectF(cx - w/2, cy - h/2, w, h);
                if (classId >= 0 && classId < m_labelModel->rowCount())
                    box.label = m_labelModel->item(classId, 0)->text();
                else
                    box.label = "unknown";
                boxes.append(box);
            }
            f.close();
        }
    }

    m_annotView->setBoxes(boxes);
    statusBar()->showMessage("已加载：" + QFileInfo(path).fileName(), 2000);
}

void MainWindow::annotSaveAnnotations()
{
    if (m_annotView->getImage().isNull()) return;
    QVector<BoundingBox> boxes = m_annotView->getBoxes();
    QString imgPath = ""; // track current

    if (boxes.isEmpty()) return;

    // 找到当前图片路径
    if (m_annotCurrentIndex >= 0 && m_annotCurrentIndex < m_annotImageFiles.size()) {
        imgPath = m_annotImageFiles[m_annotCurrentIndex];
    }
    if (imgPath.isEmpty()) return;

    QImage img = m_annotView->getImage();
    int w = img.width(), h = img.height();

    QString labelPath = annotImagePathToLabelPath(imgPath);
    QFile f(labelPath);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) return;
    QTextStream ts(&f);
    ts.setEncoding(QStringConverter::Utf8);

    // 标签名 -> ID 映射
    QMap<QString, int> labelToId;
    for (int i = 0; i < m_labelModel->rowCount(); ++i)
        labelToId[m_labelModel->item(i, 0)->text()] = i;

    for (const auto &box : boxes) {
        int classId = labelToId.value(box.label, 0);
        double cx = box.rect.center().x();
        double cy = box.rect.center().y();
        double bw = box.rect.width();
        double bh = box.rect.height();
        ts << QString::number(classId) << " "
           << QString::number(cx, 'g', 6) << " "
           << QString::number(cy, 'g', 6) << " "
           << QString::number(bw, 'g', 6) << " "
           << QString::number(bh, 'g', 6) << "\n";
    }
    f.close();
    statusBar()->showMessage("已保存标注：" + labelPath, 2000);
}

// ============================================================
// 标注槽函数
// ============================================================

void MainWindow::on_btn_annotOpenFolder_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(this, "选择图片文件夹");
    if (dir.isEmpty()) return;

    m_annotCurrentDir = dir;
    QDir d(dir);
    QStringList filters = {"*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"};
    m_annotImageFiles = d.entryList(filters, QDir::Files | QDir::Readable, QDir::Name);

    if (m_annotImageFiles.isEmpty()) {
        QMessageBox::information(this, "提示", "该文件夹中没有找到图片文件。");
        return;
    }

    // 补全完整路径
    for (int i = 0; i < m_annotImageFiles.size(); ++i)
        m_annotImageFiles[i] = dir + "/" + m_annotImageFiles[i];

    // 填充列表
    ui->listWidget_images->clear();
    for (const QString &f : m_annotImageFiles)
        ui->listWidget_images->addItem(QFileInfo(f).fileName());

    m_annotCurrentIndex = 0;
    annotLoadImage(m_annotImageFiles[0]);
}

void MainWindow::on_listWidget_images_itemClicked(QListWidgetItem *item)
{
    if (!item) return;
    QString fileName = item->text();
    QString fullPath = m_annotCurrentDir + "/" + fileName;
    for (int i = 0; i < m_annotImageFiles.size(); ++i) {
        if (m_annotImageFiles[i] == fullPath) {
            m_annotCurrentIndex = i;
            annotLoadImage(fullPath);
            break;
        }
    }
}

void MainWindow::on_btn_annotPrev_clicked()
{
    if (m_annotCurrentIndex > 0) {
        --m_annotCurrentIndex;
        annotLoadImage(m_annotImageFiles[m_annotCurrentIndex]);
        ui->listWidget_images->setCurrentRow(m_annotCurrentIndex);
    }
}

void MainWindow::on_btn_annotNext_clicked()
{
    if (m_annotCurrentIndex < m_annotImageFiles.size() - 1) {
        ++m_annotCurrentIndex;
        annotLoadImage(m_annotImageFiles[m_annotCurrentIndex]);
        ui->listWidget_images->setCurrentRow(m_annotCurrentIndex);
    }
}

void MainWindow::on_btn_annotAddLabel_clicked()
{
    bool ok = false;
    QString name = QInputDialog::getText(this, "添加标签", "标签名称：", QLineEdit::Normal, "", &ok);
    if (!ok || name.trimmed().isEmpty()) return;
    name = name.trimmed();

    // 检查重复
    for (int i = 0; i < m_labelModel->rowCount(); ++i) {
        if (m_labelModel->item(i, 0)->text() == name) {
            QMessageBox::information(this, "提示", "标签 '" + name + "' 已存在。");
            return;
        }
    }

    int row = m_labelModel->rowCount();
    m_labelModel->insertRow(row);
    m_labelModel->setItem(row, 0, new QStandardItem(name));
    m_labelModel->setItem(row, 1, new QStandardItem("■"));
    m_labelModel->setItem(row, 2, new QStandardItem("✓"));
    annotSyncLabels();
}

void MainWindow::on_btn_annotDelLabel_clicked()
{
    QModelIndex idx = ui->tableView_labels->currentIndex();
    if (!idx.isValid()) return;
    int row = idx.row();
    QString name = m_labelModel->item(row, 0)->text();
    m_labelModel->removeRow(row);
    m_annotView->removeLabel(name);
    annotSyncLabels();
}

void MainWindow::on_btn_annotSave_clicked()
{
    annotSaveAnnotations();
}

void MainWindow::on_annotBoxesChanged(const QVector<BoundingBox> &)
{
    // 框变化时自动设置当前标签
    annotSyncLabels();
}

void MainWindow::on_annotationLabelChanged(const QString &label)
{
    Q_UNUSED(label);
}

void MainWindow::on_btn_annotSelectModel_clicked()
{
    QString file = QFileDialog::getOpenFileName(
        this, "选择 YOLO 模型文件（best.pt）",
        m_annotCurrentDir,
        "Model Files (*.pt *.pth);;All Files (*)"
    );
    if (file.isEmpty()) return;

    m_yoloModelPath = file;
    ui->label_modelPath->setText(QFileInfo(file).fileName());
    ui->label_modelPath->setToolTip(file);
}

void MainWindow::on_btn_annotAutoLabel_clicked()
{
    if (m_annotView->getImage().isNull()) {
        QMessageBox::information(this, "提示", "请先打开一张图片。");
        return;
    }
    if (m_yoloModelPath.isEmpty()) {
        QMessageBox::information(this, "提示", "请先点击「选择模型文件」选择 best.pt 模型。");
        return;
    }
    if (!QFile::exists(m_yoloModelPath)) {
        QMessageBox::warning(this, "错误", "模型文件不存在：" + m_yoloModelPath);
        return;
    }

    QString python = ui->lineEdit_python->text().trimmed();
    if (python.isEmpty()) python = "python";

    // 找到当前图片
    if (m_annotCurrentIndex < 0 || m_annotCurrentIndex >= m_annotImageFiles.size()) return;
    QString imgPath = m_annotImageFiles[m_annotCurrentIndex];

    QString scriptPath = "D:/YC/software/QT/PuddleDetection/auto_label.py";
    QString outPath = QFileInfo(imgPath).absolutePath() + "/.auto_label_temp.json";

    m_autoProcess->setProcessChannelMode(QProcess::MergedChannels);
    m_autoProcess->start(python, {scriptPath, imgPath, m_yoloModelPath, outPath});

    ui->btn_annotAutoLabel->setEnabled(false);
    statusBar()->showMessage("正在进行 YOLO 自动标注...", 3000);
}

void MainWindow::on_annotAutoReadyRead()
{
    QByteArray raw = m_autoProcess->readAllStandardOutput();
    QString text = QString::fromUtf8(raw);
    static const QRegularExpression ansiRe("\\x1b\\[[^a-zA-Z]*[a-zA-Z]");
    text.remove(ansiRe);
    statusBar()->showMessage(text.trimmed(), 3000);
}

void MainWindow::on_annotAutoFinished(int exitCode, QProcess::ExitStatus)
{
    ui->btn_annotAutoLabel->setEnabled(true);

    if (exitCode != 0) {
        QMessageBox::warning(this, "自动标注失败",
            "YOLO 推理失败，退出码：" + QString::number(exitCode));
        return;
    }

    // 找到当前图片
    if (m_annotCurrentIndex < 0 || m_annotCurrentIndex >= m_annotImageFiles.size()) return;
    QString imgPath = m_annotImageFiles[m_annotCurrentIndex];
    QString jsonPath = QFileInfo(imgPath).absolutePath() + "/.auto_label_temp.json";

    if (!QFile::exists(jsonPath)) {
        statusBar()->showMessage("未找到推理结果文件。", 3000);
        return;
    }

    // 解析 JSON 并加载结果
    QFile f(jsonPath);
    if (!f.open(QIODevice::ReadOnly)) return;
    QByteArray data = f.readAll();
    f.close();

    QJsonDocument doc = QJsonDocument::fromJson(data);
    if (!doc.isArray()) return;
    QJsonArray arr = doc.array();

    QVector<BoundingBox> boxes;
    QImage img = m_annotView->getImage();
    if (img.isNull()) return;
    int imgW = img.width(), imgH = img.height();

    for (int i = 0; i < arr.size(); ++i) {
        QJsonObject obj = arr[i].toObject();
        if (obj.isEmpty()) continue;
        BoundingBox box;
        box.id = i;
        box.label = obj.value("label").toString();
        QJsonArray xywh = obj.value("box").toArray();
        if (xywh.size() < 4) continue;
        // auto_label.py 输出的 [cx,cy,w,h] 已经是像素坐标，无需再归一化
        // 直接转换为 QRectF(x,y,w,h)
        double cx = xywh[0].toDouble();
        double cy = xywh[1].toDouble();
        double bw = xywh[2].toDouble();
        double bh = xywh[3].toDouble();
        box.rect = QRectF(cx - bw/2, cy - bh/2, bw, bh);
        boxes.append(box);
    }

    m_annotView->setBoxes(boxes);
    QFile::remove(jsonPath);
    statusBar()->showMessage(QString("自动标注完成，找到 %1 个目标。").arg(boxes.size()), 5000);
}

// ============================================================
// ============================================================
// ===== 模型量化相关槽函数 =====================================
// ============================================================
// ============================================================

// 选择源模型（.pt）
void MainWindow::on_btn_quantSelectModel_clicked()
{
    QString file = QFileDialog::getOpenFileName(
        this, "选择训练好的模型文件 (.pt)",
        "D:/YC/software/QT/PuddleDetection",
        "PyTorch Model (*.pt *.pth);;All Files (*)"
    );
    if (!file.isEmpty()) {
        ui->lineEdit_quantModel->setText(file);
        appendQuantLog("📦 源模型已选择：" + file);
    }
}

// 选择校准图片目录（静态量化使用）
void MainWindow::on_btn_quantSelectCalib_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(
        this, "选择校准图片目录（静态量化时使用）",
        "D:/YC/software/QT/PuddleDetection/dataset"
    );
    if (!dir.isEmpty()) {
        ui->lineEdit_quantCalib->setText(dir);
        appendQuantLog("🖼 校准图片目录：" + dir);
    }
}

// 选择输出目录
void MainWindow::on_btn_quantSelectOutput_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(
        this, "选择量化模型输出目录",
        "D:/YC/software/QT/PuddleDetection"
    );
    if (!dir.isEmpty()) {
        ui->lineEdit_quantOutput->setText(dir);
    }
}

// 开始量化
void MainWindow::on_btn_quantStart_clicked()
{
    // ---- 参数检查 ----
    QString modelPt = ui->lineEdit_quantModel->text().trimmed();
    if (modelPt.isEmpty() || !QFile::exists(modelPt)) {
        QMessageBox::warning(this, "错误", "请先选择有效的源模型文件（.pt）！");
        return;
    }

    QString outputDir = ui->lineEdit_quantOutput->text().trimmed();
    if (outputDir.isEmpty()) {
        outputDir = "D:/YC/software/QT/PuddleDetection/quant_models";
        ui->lineEdit_quantOutput->setText(outputDir);
    }

    QString calibDir = ui->lineEdit_quantCalib->text().trimmed();
    if (calibDir.isEmpty()) calibDir = outputDir; // 占位，脚本会判断是否存在

    QString quantTypeRaw = ui->combo_quantType->currentText();
    QString quantType = quantTypeRaw.startsWith("static") ? "static" : "dynamic";

    QString python = ui->lineEdit_python->text().trimmed();
    if (python.isEmpty()) python = "python";

    QString scriptPath = "D:/YC/software/QT/PuddleDetection/quantize.py";

    // ---- 重置状态 ----
    m_quantOutputPath.clear();
    setQuantState(true);
    ui->text_quantLog->clear();

    appendQuantLog("═══════════════════════════════════════════");
    appendQuantLog("  水坑检测 — 模型量化工具  开始");
    appendQuantLog("═══════════════════════════════════════════");
    appendQuantLog(QString("📅 时间   : %1").arg(QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss")));
    appendQuantLog(QString("📦 源模型 : %1").arg(modelPt));
    appendQuantLog(QString("📁 输出   : %1").arg(outputDir));
    appendQuantLog(QString("⚙️  类型   : %1 INT8").arg(quantType.toUpper()));
    appendQuantLog("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    QStringList args;
    args << scriptPath << modelPt << outputDir << calibDir << quantType;
    m_quantProcess->start(python, args);

    if (!m_quantProcess->waitForStarted(5000)) {
        appendQuantLog("❌ 量化进程启动失败！请检查 Python 路径。");
        setQuantState(false);
    }
}

// 停止量化
void MainWindow::on_btn_quantStop_clicked()
{
    if (m_quantProcess->state() != QProcess::NotRunning) {
        m_quantProcess->kill();
        m_quantProcess->waitForFinished(3000);
        appendQuantLog("⏹ 量化已被手动停止。");
    }
    setQuantState(false);
}

// 清空量化日志
void MainWindow::on_btn_quantClearLog_clicked()
{
    ui->text_quantLog->clear();
}

// 量化进程实时输出
void MainWindow::onQuantReadyRead()
{
    QByteArray raw = m_quantProcess->readAllStandardOutput();
    QString text = QString::fromUtf8(raw);
    static const QRegularExpression ansiRe("\\x1b\\[[^a-zA-Z]*[a-zA-Z]");
    text.remove(ansiRe);
    text.replace("\r\n", "\n");
    text.replace("\r", "\n");

    QStringList lines = text.split('\n');
    for (const QString &line : lines) {
        QString trimmed = line.trimmed();
        if (trimmed.isEmpty()) continue;
        appendQuantLog(trimmed);

        // 捕捉量化输出路径（脚本特殊标记行）
        if (trimmed.startsWith("QUANT_OUTPUT_PATH:")) {
            m_quantOutputPath = trimmed.mid(QString("QUANT_OUTPUT_PATH:").length()).trimmed();
        }
    }
}

// 量化进程结束
void MainWindow::onQuantFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    appendQuantLog("");
    if (exitStatus == QProcess::NormalExit && exitCode == 0) {
        appendQuantLog("════════════════ 量化完成 ════════════════");
        appendQuantLog("✅ 模型量化成功完成！");
        appendQuantLog(QString("📅 完成时间: %1").arg(
            QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss")));
        if (!m_quantOutputPath.isEmpty()) {
            appendQuantLog("💾 量化模型路径：" + m_quantOutputPath);
            // 自动填入推理模型路径
            ui->lineEdit_inferModel->setText(m_quantOutputPath);
            m_inferModelPath = m_quantOutputPath;
            appendQuantLog("✅ 推理模型路径已自动填入，可直接点击「运行推理」！");
        }
        appendQuantLog("═══════════════════════════════════════");
        ui->label_quantStatus->setText("量化完成 ✅");
        ui->label_quantStatus->setStyleSheet("color: #4CAF50; font-weight: bold;");
        ui->statusbar->showMessage("模型量化完成！", 0);

        // ---- 写入数据库 ----
        if (m_db->isOpen()) {
            QString srcModel = ui->lineEdit_quantModel->text().trimmed();
            QString qtype    = ui->combo_quantType->currentText().startsWith("static")
                               ? "static" : "dynamic";
            // 计算文件大小（MB）
            auto fileMB = [](const QString &path) -> double {
                QFileInfo fi(path);
                return fi.exists() ? fi.size() / 1048576.0 : 0.0;
            };
            double origMB  = fileMB(srcModel);
            double quantMB = fileMB(m_quantOutputPath);
            qint64 rid = m_db->insertQuantRecord(srcModel, qtype,
                                                 m_quantOutputPath,
                                                 origMB, quantMB,
                                                 "completed");
            if (rid > 0) {
                appendQuantLog(QString("🗄  量化记录已保存到数据库（ID=%1）").arg(rid));
                dbRefreshStats();
            }
        }
    } else {
        appendQuantLog("════════════════ 量化异常 ════════════════");
        appendQuantLog(QString("❌ 退出码 %1（%2）")
                       .arg(exitCode)
                       .arg(exitStatus == QProcess::CrashExit ? "进程崩溃" : "异常退出"));
        appendQuantLog("请查看上方日志中的错误信息。");
        appendQuantLog("常见原因：未安装 onnxruntime / onnx / ultralytics");
        appendQuantLog("解决：pip install onnxruntime onnx ultralytics");
        appendQuantLog("═══════════════════════════════════════");
        ui->label_quantStatus->setText("量化失败 ❌");
        ui->label_quantStatus->setStyleSheet("color: #f44336; font-weight: bold;");

        // ---- 写入失败记录 ----
        if (m_db->isOpen()) {
            QString srcModel = ui->lineEdit_quantModel->text().trimmed();
            QString qtype    = ui->combo_quantType->currentText().startsWith("static")
                               ? "static" : "dynamic";
            m_db->insertQuantRecord(srcModel, qtype, QString(), 0.0, 0.0, "failed");
        }
    }
    setQuantState(false);
}

// 量化进程错误
void MainWindow::onQuantError(QProcess::ProcessError error)
{
    QString errMsg;
    switch (error) {
        case QProcess::FailedToStart: errMsg = "进程启动失败（Python 路径错误？）"; break;
        case QProcess::Crashed:       errMsg = "进程崩溃";                         break;
        default:                      errMsg = "未知错误";                         break;
    }
    appendQuantLog("❌ 量化进程错误：" + errMsg);
    setQuantState(false);
}

// ============================================================
// ===== 量化后推理相关槽函数 ====================================
// ============================================================

// 手动选择量化 ONNX 模型
void MainWindow::on_btn_inferSelectModel_clicked()
{
    QString file = QFileDialog::getOpenFileName(
        this, "选择量化后的 ONNX 模型",
        "D:/YC/software/QT/PuddleDetection/quant_models",
        "ONNX Model (*.onnx);;All Files (*)"
    );
    if (!file.isEmpty()) {
        m_inferModelPath = file;
        ui->lineEdit_inferModel->setText(file);
        appendQuantLog("🤖 推理模型已选择：" + QFileInfo(file).fileName());
    }
}

// 选择推理图片
void MainWindow::on_btn_inferSelectImage_clicked()
{
    QString file = QFileDialog::getOpenFileName(
        this, "选择推理图片",
        m_inferModelPath.isEmpty() ? "D:/YC" : QFileInfo(m_inferModelPath).absolutePath(),
        "Images (*.jpg *.jpeg *.png *.bmp *.webp);;All Files (*)"
    );
    if (!file.isEmpty()) {
        m_inferImagePath = file;
        ui->lineEdit_inferImage->setText(file);
        // 预览原图
        QPixmap pix(file);
        if (!pix.isNull()) {
            ui->label_inferImage->setPixmap(
                pix.scaled(ui->label_inferImage->size(),
                           Qt::KeepAspectRatio,
                           Qt::SmoothTransformation)
            );
        }
        appendQuantLog("🖼 推理图片：" + QFileInfo(file).fileName());
    }
}

// 运行推理
void MainWindow::on_btn_inferRun_clicked()
{
    // ---- 参数检查 ----
    if (m_inferModelPath.isEmpty() || !QFile::exists(m_inferModelPath)) {
        QMessageBox::warning(this, "错误",
            "请先选择量化后的 ONNX 模型文件！\n"
            "（先完成量化，或手动点击「浏览」选择 .onnx 文件）");
        return;
    }
    if (m_inferImagePath.isEmpty() || !QFile::exists(m_inferImagePath)) {
        QMessageBox::warning(this, "错误", "请先选择推理图片！");
        return;
    }

    // ---- 检查推理引擎选择 ----
    int engineIdx = ui->combo_inferEngine->currentIndex();
    if (engineIdx == 1) {
        // C++ 推理
        runCppInference();
        return;
    }

    // ---- Python 推理 ----
    QString python = ui->lineEdit_python->text().trimmed();
    if (python.isEmpty()) python = "python";

    QString scriptPath = "D:/YC/software/QT/PuddleDetection/infer_quant.py";
    QString outJson = QFileInfo(m_inferImagePath).absolutePath() + "/.infer_result.json";

    double conf = ui->spin_inferConf->value();
    double iou  = ui->spin_inferIou->value();

    appendQuantLog("");
    appendQuantLog("════════════ 推理开始（Python） ════════════");
    appendQuantLog(QString("📅 %1").arg(QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss")));
    appendQuantLog("🤖 模型：" + QFileInfo(m_inferModelPath).fileName());
    appendQuantLog("🖼 图片：" + QFileInfo(m_inferImagePath).fileName());

    ui->btn_inferRun->setEnabled(false);
    ui->label_inferResult->setText("推理中... ⏳");
    ui->label_inferResult->setStyleSheet("color: #FF9800; font-weight: bold;");
    ui->text_inferBoxes->clear();

    QStringList args;
    args << scriptPath
         << m_inferModelPath
         << m_inferImagePath
         << outJson
         << QString::number(conf, 'f', 2)
         << QString::number(iou,  'f', 2);

    m_inferProcess->start(python, args);
    if (!m_inferProcess->waitForStarted(5000)) {
        appendQuantLog("❌ 推理进程启动失败！");
        ui->btn_inferRun->setEnabled(true);
        ui->label_inferResult->setText("推理失败 ❌");
        ui->label_inferResult->setStyleSheet("color: #f44336; font-weight: bold;");
    }
}

// 推理进程实时输出
void MainWindow::onInferReadyRead()
{
    QByteArray raw = m_inferProcess->readAllStandardOutput();
    QString text = QString::fromUtf8(raw);
    static const QRegularExpression ansiRe("\\x1b\\[[^a-zA-Z]*[a-zA-Z]");
    text.remove(ansiRe);
    text.replace("\r\n", "\n");
    text.replace("\r", "\n");

    for (const QString &line : text.split('\n')) {
        QString t = line.trimmed();
        if (!t.isEmpty()) appendQuantLog(t);
    }
}

// 推理进程结束
void MainWindow::onInferFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    ui->btn_inferRun->setEnabled(true);

    if (exitStatus != QProcess::NormalExit || exitCode != 0) {
        appendQuantLog("❌ 推理失败，退出码：" + QString::number(exitCode));
        ui->label_inferResult->setText("推理失败 ❌");
        ui->label_inferResult->setStyleSheet("color: #f44336; font-weight: bold;");
        return;
    }

    // 读取结果 JSON
    QString outJson = QFileInfo(m_inferImagePath).absolutePath() + "/.infer_result.json";
    loadInferResult(outJson);
}

// ============================================================
// ===== 量化工具函数 ============================================
// ============================================================

void MainWindow::appendQuantLog(const QString &text)
{
    ui->text_quantLog->appendPlainText(text);
    QScrollBar *sb = ui->text_quantLog->verticalScrollBar();
    sb->setValue(sb->maximum());
}

void MainWindow::setQuantState(bool running)
{
    ui->btn_quantStart->setEnabled(!running);
    ui->btn_quantStop->setEnabled(running);
    ui->btn_quantSelectModel->setEnabled(!running);
    ui->btn_quantSelectCalib->setEnabled(!running);
    ui->btn_quantSelectOutput->setEnabled(!running);
    ui->combo_quantType->setEnabled(!running);

    if (running) {
        // busy 模式：无限进度条
        ui->progressBar_quant->setRange(0, 0);
        ui->label_quantStatus->setText("量化中... ⏳");
        ui->label_quantStatus->setStyleSheet("color: #FF9800; font-weight: bold;");
    } else {
        // 停止 busy 模式
        ui->progressBar_quant->setRange(0, 100);
        ui->progressBar_quant->setValue(
            ui->label_quantStatus->text().contains("完成") ? 100 : 0
        );
        if (ui->label_quantStatus->text().startsWith("量化中")) {
            ui->label_quantStatus->setText("等待开始");
            ui->label_quantStatus->setStyleSheet("color: #888; font-weight: bold;");
        }
    }
}

void MainWindow::loadInferResult(const QString &jsonPath)
{
    if (!QFile::exists(jsonPath)) {
        appendQuantLog("⚠️ 未找到推理结果文件。");
        ui->label_inferResult->setText("无结果文件 ⚠️");
        return;
    }

    QFile f(jsonPath);
    if (!f.open(QIODevice::ReadOnly)) return;
    QByteArray data = f.readAll();
    f.close();
    QFile::remove(jsonPath);

    QJsonDocument doc = QJsonDocument::fromJson(data);
    if (!doc.isArray()) {
        appendQuantLog("⚠️ 结果 JSON 格式错误。");
        return;
    }
    QJsonArray arr = doc.array();

    // 更新结果标签
    ui->label_inferResult->setText(
        QString("检测到 %1 个目标 ✅").arg(arr.size())
    );
    ui->label_inferResult->setStyleSheet("color: #4CAF50; font-weight: bold;");

    // 显示带框图像
    showInferImage(m_inferImagePath, arr);

    // 文字详情 + 统计卡片
    ui->text_inferBoxes->clear();
    double maxConf = 0.0, sumConf = 0.0;
    int    count   = arr.size();
    if (arr.isEmpty()) {
        ui->text_inferBoxes->appendPlainText("⚠️ 未检测到目标");
    } else {
        for (int i = 0; i < arr.size(); ++i) {
            QJsonObject obj = arr[i].toObject();
            QString label = obj["label"].toString();
            double  conf  = obj["conf"].toDouble();
            if (conf > maxConf) maxConf = conf;
            sumConf += conf;
            QJsonArray box = obj["box"].toArray();
            ui->text_inferBoxes->appendPlainText(
                QString("[%1] %2  conf=%.3f  box=[%3,%4,%5,%6]")
                .arg(i + 1)
                .arg(label)
                .arg(conf)
                .arg((int)box[0].toDouble()).arg((int)box[1].toDouble())
                .arg((int)box[2].toDouble()).arg((int)box[3].toDouble())
            );
        }
    }

    // 更新三个统计卡片
    double avgConf = (count > 0) ? sumConf / count : 0.0;
    ui->label_inferCountVal->setText(count > 0 ? QString::number(count) : "0");
    ui->label_inferMaxConfVal->setText(count > 0 ? QString::number(maxConf, 'f', 3) : "--");
    ui->label_inferAvgConfVal->setText(count > 0 ? QString::number(avgConf, 'f', 3) : "--");

    appendQuantLog(QString("════════════ 推理完成：检测到 %1 个目标 ✅ ════════════").arg(arr.size()));

    // ---- 写入数据库 ----
    if (m_db->isOpen()) {
        // 将检测结果序列化为 JSON 字符串存入 DB
        QJsonDocument detectDoc(arr);
        QString detectJson = QString::fromUtf8(detectDoc.toJson(QJsonDocument::Compact));
        qint64 rid = m_db->insertInferResult(
            m_inferModelPath,
            m_inferImagePath,
            arr.size(),
            maxConf,
            0.0,   // infer time 暂不计时
            detectJson
        );
        if (rid > 0) {
            appendQuantLog(QString("🗄  推理结果已保存到数据库（ID=%1）").arg(rid));
            dbRefreshStats();
        }
    }
}

// ============================================================
// C++ ONNX Runtime 推理（Feature 6.5）
// ============================================================
void MainWindow::runCppInference()
{
    double conf = ui->spin_inferConf->value();
    double iou  = ui->spin_inferIou->value();

    appendQuantLog("");
    appendQuantLog("════════════ 推理开始（C++） ════════════");
    appendQuantLog(QString("📅 %1").arg(QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss")));
    appendQuantLog("🤖 模型：" + QFileInfo(m_inferModelPath).fileName());
    appendQuantLog("🖼 图片：" + QFileInfo(m_inferImagePath).fileName());
    appendQuantLog("⚙️  引擎：C++ onnxruntime");

    ui->btn_inferRun->setEnabled(false);
    ui->label_inferResult->setText("推理中... ⏳ (C++)");
    ui->label_inferResult->setStyleSheet("color: #FF9800; font-weight: bold;");
    ui->text_inferBoxes->clear();

    // 加载模型（如果未加载或路径不同）
    if (!m_cppInfer.isLoaded()) {
        appendQuantLog("📦 加载 ONNX 模型...");
        if (!m_cppInfer.loadModel(m_inferModelPath)) {
            appendQuantLog("❌ 模型加载失败：" + m_cppInfer.lastError());
            ui->label_inferResult->setText("模型加载失败 ❌");
            ui->label_inferResult->setStyleSheet("color: #f44336; font-weight: bold;");
            ui->btn_inferRun->setEnabled(true);
            return;
        }
        appendQuantLog("✅ 模型加载成功");
        appendQuantLog(QString("   输入尺寸：%1×%2  类别数：%3")
                       .arg(m_cppInfer.inputWidth())
                       .arg(m_cppInfer.inputHeight())
                       .arg(m_cppInfer.classNames().size()));
    }

    // 执行推理
    appendQuantLog("🔍 执行推理...");
    QJsonArray results = m_cppInfer.inference(m_inferImagePath, conf, iou);

    if (!m_cppInfer.lastError().isEmpty() && results.isEmpty()) {
        appendQuantLog("❌ 推理失败：" + m_cppInfer.lastError());
        ui->label_inferResult->setText("推理失败 ❌");
        ui->label_inferResult->setStyleSheet("color: #f44336; font-weight: bold;");
        ui->btn_inferRun->setEnabled(true);
        return;
    }

    int count = results.size();
    appendQuantLog(QString("════════════ 推理完成：检测到 %1 个目标 ✅ ════════════").arg(count));

    // 更新结果标签
    ui->label_inferResult->setText(
        QString("检测到 %1 个目标 ✅ (C++)").arg(count));
    ui->label_inferResult->setStyleSheet("color: #4CAF50; font-weight: bold;");

    // 显示带框图像
    showInferImage(m_inferImagePath, results);

    // 文字详情 + 统计卡片
    double maxConf = 0.0, sumConf = 0.0;
    if (results.isEmpty()) {
        ui->text_inferBoxes->appendPlainText("⚠️ 未检测到目标");
    } else {
        for (int i = 0; i < results.size(); ++i) {
            QJsonObject obj = results[i].toObject();
            QString label = obj["label"].toString();
            double  c     = obj["conf"].toDouble();
            if (c > maxConf) maxConf = c;
            sumConf += c;
            QJsonArray box = obj["box"].toArray();
            ui->text_inferBoxes->appendPlainText(
                QString("[%1] %2  conf=%.3f  box=[%3,%4,%5,%6]")
                .arg(i + 1)
                .arg(label)
                .arg(c)
                .arg((int)box[0].toDouble()).arg((int)box[1].toDouble())
                .arg((int)box[2].toDouble()).arg((int)box[3].toDouble())
            );
        }
    }

    // 更新三个统计卡片
    double avgConf = (count > 0) ? sumConf / count : 0.0;
    ui->label_inferCountVal->setText(count > 0 ? QString::number(count) : "0");
    ui->label_inferMaxConfVal->setText(count > 0 ? QString::number(maxConf, 'f', 3) : "--");
    ui->label_inferAvgConfVal->setText(count > 0 ? QString::number(avgConf, 'f', 3) : "--");

    // ---- 写入数据库 ----
    if (m_db->isOpen()) {
        QJsonDocument detectDoc(results);
        QString detectJson = QString::fromUtf8(detectDoc.toJson(QJsonDocument::Compact));
        qint64 rid = m_db->insertInferResult(
            m_inferModelPath,
            m_inferImagePath,
            count,
            maxConf,
            0.0,
            detectJson
        );
        if (rid > 0) {
            appendQuantLog(QString("🗄  推理结果已保存到数据库（ID=%1）").arg(rid));
            dbRefreshStats();
        }
    }

    ui->btn_inferRun->setEnabled(true);
}

void MainWindow::showInferImage(const QString &imagePath, const QJsonArray &detections)
{
    QImage img(imagePath);
    if (img.isNull()) {
        ui->label_inferImage->setText("无法加载图片");
        return;
    }

    // 在图上绘制检测框
    if (!detections.isEmpty()) {
        QPainter painter(&img);
        painter.setRenderHint(QPainter::Antialiasing);

        const QList<QColor> colors = {
            QColor("#FF6B6B"), QColor("#4ECDC4"), QColor("#45B7D1"),
            QColor("#F7DC6F"), QColor("#BB8FCE"), QColor("#98D8C8"),
        };

        QFont font = painter.font();
        font.setPixelSize(qMax(14, img.height() / 30));
        font.setBold(true);
        painter.setFont(font);

        for (int i = 0; i < detections.size(); ++i) {
            QJsonObject obj = detections[i].toObject();
            QJsonArray  box = obj["box"].toArray();
            QString   label = obj["label"].toString();
            double     conf = obj["conf"].toDouble();
            if (box.size() < 4) continue;

            int x = (int)box[0].toDouble();
            int y = (int)box[1].toDouble();
            int w = (int)box[2].toDouble();
            int h = (int)box[3].toDouble();

            QColor color = colors[i % colors.size()];

            // 绘制边框
            QPen pen(color, qMax(2, img.width() / 300));
            painter.setPen(pen);
            painter.drawRect(x, y, w, h);

            // 绘制标签背景
            QString text = QString("%1 %.2f").arg(label).arg(conf);
            QFontMetrics fm(font);
            QRect textRect = fm.boundingRect(text);
            int tx = x, ty = qMax(0, y - textRect.height() - 4);
            painter.fillRect(tx, ty, textRect.width() + 8, textRect.height() + 4, color);

            // 绘制标签文字
            painter.setPen(Qt::white);
            painter.drawText(tx + 4, ty + textRect.height(), text);
        }
        painter.end();
    }

    // 显示在 label 上
    QPixmap pix = QPixmap::fromImage(img);
    ui->label_inferImage->setPixmap(
        pix.scaled(ui->label_inferImage->size(),
                   Qt::KeepAspectRatio,
                   Qt::SmoothTransformation)
    );
}

// ============================================================
// ============================================================
// ===== 数据库与可视化 ==========================================
// ============================================================
// ============================================================

// ============================================================
// 标注信息列表 + 改选中框类别
// ============================================================
void MainWindow::updateAnnotInfoList()
{
    if (!ui || !ui->listWidget_annotInfo) return;
    ui->listWidget_annotInfo->clear();

    QVector<BoundingBox> boxes = m_annotView->getBoxes();
    if (boxes.isEmpty()) {
        ui->listWidget_annotInfo->addItem("（当前图片没有标注框）");
        return;
    }

    int selectedId = m_annotView->getSelectedId();
    int imgW = 0, imgH = 0;
    QImage img;
    // 用当前正在显示的图片路径反推尺寸
    QString curPath;
    if (m_annotCurrentIndex >= 0 && m_annotCurrentIndex < m_annotImageFiles.size()) {
        curPath = m_annotImageFiles.at(m_annotCurrentIndex);
    }
    if (!curPath.isEmpty()) {
        QPixmap pix(curPath);
        if (!pix.isNull()) {
            imgW = pix.width();
            imgH = pix.height();
        }
    }
    // 兜底：图片 640x640
    if (imgW <= 0) imgW = 640;
    if (imgH <= 0) imgH = 640;

    for (int i = 0; i < boxes.size(); ++i) {
        const BoundingBox &b = boxes[i];
        double nx = b.rect.x();
        double ny = b.rect.y();
        double nw = b.rect.width();
        double nh = b.rect.height();
        int    px = int(nx * imgW);
        int    py = int(ny * imgH);
        int    pw = int(nw * imgW);
        int    ph = int(nh * imgH);
        QString label = b.label.isEmpty() ? "(未分类)" : b.label;
        QString star  = (b.id == selectedId) ? "★ " : "  ";
        QString line  = QString("%1#%2  %3\n      归一化:(%.3f, %.3f, %.3f, %.3f)\n      像素:(%4,%5,%6×%7)")
                          .arg(star)
                          .arg(b.id)
                          .arg(label)
                          .arg(nx).arg(ny).arg(nw).arg(nh)
                          .arg(px).arg(py).arg(pw).arg(ph);
        QListWidgetItem *item = new QListWidgetItem(line, ui->listWidget_annotInfo);
        if (b.id == selectedId) {
            item->setBackground(QColor(255, 255, 200));
        }
        ui->listWidget_annotInfo->addItem(item);
    }
}

void MainWindow::on_annotSelectionChanged()
{
    updateAnnotInfoList();
}

void MainWindow::on_tableView_labels_clicked(const QModelIndex &index)
{
    if (!index.isValid()) return;
    QString label = m_labelModel->item(index.row(), 0)->text();
    // 1) 同步"当前标签" → 后续新画的框会用这个 label
    m_annotView->setCurrentLabel(label);
    // 2) 尝试改"已选中的框"为这个 label
    if (m_annotView->getSelectedId() >= 0) {
        if (m_annotView->changeSelectedBoxLabel(label)) {
            appendLog(QString("✅ 选中框 #%1 的类别已改为：%2")
                          .arg(m_annotView->getSelectedId()).arg(label));
        }
    }
    // 刷新一下信息列表（标签色块没变，但类别名字要刷新）
    updateAnnotInfoList();
}

void MainWindow::on_btn_annotChangeLabel_clicked()
{
    // 把 tableView_labels 当前选中的标签，作为"新类别"应用给选中的框
    QModelIndex cur = ui->tableView_labels->currentIndex();
    if (!cur.isValid()) {
        QMessageBox::information(this, "提示", "请先在右侧「标签管理」里选中一个目标标签。");
        return;
    }
    QString label = m_labelModel->item(cur.row(), 0)->text();
    if (m_annotView->getSelectedId() < 0) {
        QMessageBox::information(this, "提示",
            "请先在画布上点击要修改的标注框（点中后会高亮）。");
        return;
    }
    if (m_annotView->changeSelectedBoxLabel(label)) {
        appendLog(QString("✅ 选中框 #%1 的类别已改为：%2")
                      .arg(m_annotView->getSelectedId()).arg(label));
        updateAnnotInfoList();
    } else {
        appendLog(QString("⚠️ 该框的类别已经是：%1").arg(label));
    }
}

void MainWindow::dbInit()
{
    // 数据库放在可执行文件同目录下（可移植，不依赖硬编码路径）
    QString dbPath = QCoreApplication::applicationDirPath() + "/puddle_detection.db";
    qDebug() << "[MainWindow] DB path:" << dbPath;
    if (!m_db->init(dbPath)) {
        qWarning() << "[MainWindow] DB init failed:" << m_db->lastError();
        ui->statusbar->showMessage("⚠️ 数据库初始化失败：" + m_db->lastError(), 0); // 0=永久显示
    } else {
        qDebug() << "[MainWindow] DB opened:" << dbPath;
        ui->statusbar->showMessage("数据库：" + dbPath, 8000);
        // 刷新概览统计
        dbRefreshStats();
    }
}

void MainWindow::dbRefreshStats()
{
    if (!m_db->isOpen()) {
        qDebug() << "[DB] dbRefreshStats: DB not open!";
        return;
    }

    // 打印数据库原始记录数，定位读不到数据的问题
    m_db->debugPrintAllCounts();

    int trainCnt = m_db->countTrainRecords();
    int annotCnt = m_db->countAnnotationStats();
    int quantCnt = m_db->countQuantRecords();
    int inferCnt = m_db->countInferResults();
    double avgMap  = m_db->avgTrainMap50();
    double avgConf = m_db->avgInferConfidence();

    qDebug() << "[DB] dbRefreshStats result:"
             << "train=" << trainCnt
             << "annot=" << annotCnt
             << "quant=" << quantCnt
             << "infer=" << inferCnt;

    ui->label_statTrainCount->setText(QString::number(trainCnt));
    ui->label_statAnnotCount->setText(QString::number(annotCnt));
    ui->label_statQuantCount->setText(QString::number(quantCnt));
    ui->label_statInferCount->setText(QString::number(inferCnt));

    if (avgMap > 0)
        ui->label_statAvgMap->setText(QString::number(avgMap, 'f', 3));
    else
        ui->label_statAvgMap->setText("N/A");

    if (avgConf > 0)
        ui->label_statAvgConf->setText(QString::number(avgConf, 'f', 3));
    else
        ui->label_statAvgConf->setText("N/A");
}

void MainWindow::dbLoadTable(int tableIndex)
{
    if (!m_db->isOpen()) return;

    QStringList headers;
    QSqlQuery q(QSqlDatabase::database());  // 占位，下面各case会重新赋值

    switch (tableIndex) {
    case 0: { // 训练记录
        QString status = ui->combo_dbStatus->currentText();
        q = m_db->queryTrainRecords(status == "全部" ? QString() : status);

        QTableWidget *tw = ui->tableWidget_dbResult;
        tw->clear();
        tw->setColumnCount(8);
        tw->setHorizontalHeaderLabels({"时间", "模型", "Epoch", "状态", "mAP50", "Precision", "Recall", "模型文件"});
        tw->setRowCount(0);
        tw->setSortingEnabled(false); // 先关排序再填数据

        // 列宽
        tw->setColumnWidth(0, 150);  // 时间
        tw->setColumnWidth(1, 120);  // 模型
        tw->setColumnWidth(2, 80);   // Epoch
        tw->setColumnWidth(3, 100);  // 状态
        tw->setColumnWidth(4, 100);  // mAP50
        tw->setColumnWidth(5, 100);  // Precision
        tw->setColumnWidth(6, 100);  // Recall
        tw->setColumnWidth(7, 200);  // 模型文件
        tw->horizontalHeader()->setStretchLastSection(true);

        int row = 0;
        while (q.next()) {
            tw->insertRow(row);

            // 时间       → created_at (col 1)
            // 模型       → model_name (col 3)
            // Epoch      → epochs     (col 4)
            // 状态       → status     (col 9)
            // mAP50      → map50      (col 10)
            // Precision  → precision  (col 11)
            // Recall     → recall     (col 12)

            auto makeItem = [&](const QString &text) {
                auto *item = new QTableWidgetItem(text);
                item->setTextAlignment(Qt::AlignCenter);
                return item;
            };

            auto metricText = [](double v) -> QString {
                if (v < 0) return QStringLiteral("—");
                return QString::number(v, 'f', 4);
            };

            auto metricColor = [](double v) -> QColor {
                if (v < 0) return QColor(Qt::transparent);
                if (v > 0.7) return QColor("#4CAF50");   // 绿色
                if (v > 0.5) return QColor("#FFC107");   // 黄色
                return QColor("#F44336");                 // 红色
            };

            // 时间
            tw->setItem(row, 0, makeItem(q.value(1).toString()));
            // 模型
            tw->setItem(row, 1, makeItem(q.value(3).toString()));
            // Epoch
            tw->setItem(row, 2, makeItem(q.value(4).toString()));
            // 状态 — 上色
            {
                QString s = q.value(9).toString();
                auto *item = makeItem(s);
                if (s == "completed")      item->setForeground(QColor("#4CAF50"));
                else if (s == "failed" || s == "crashed") item->setForeground(QColor("#F44336"));
                else                       item->setForeground(QColor("#FF9800"));
                tw->setItem(row, 3, item);
            }
            // mAP50 — 颜色标注
            {
                double v = q.value(10).toDouble();
                auto *item = makeItem(metricText(v));
                item->setBackground(metricColor(v));
                if (v >= 0) item->setForeground(Qt::white);
                tw->setItem(row, 4, item);
            }
            // Precision — 颜色标注
            {
                double v = q.value(11).toDouble();
                auto *item = makeItem(metricText(v));
                item->setBackground(metricColor(v));
                if (v >= 0) item->setForeground(Qt::white);
                tw->setItem(row, 5, item);
            }
            // Recall — 颜色标注
            {
                double v = q.value(12).toDouble();
                auto *item = makeItem(metricText(v));
                item->setBackground(metricColor(v));
                if (v >= 0) item->setForeground(Qt::white);
                tw->setItem(row, 6, item);
            }
            // 模型文件 — best_pt (col 14)
            {
                QString fullPath = q.value(14).toString();
                if (fullPath.isEmpty()) {
                    tw->setItem(row, 7, makeItem("—"));
                } else {
                    auto *item = makeItem(QFileInfo(fullPath).fileName());
                    item->setToolTip(fullPath);
                    tw->setItem(row, 7, item);
                }
            }

            ++row;
        }

        // 默认按时间降序
        tw->sortByColumn(0, Qt::DescendingOrder);
        tw->setSortingEnabled(true);
        return;
    }
    case 1: { // 标注统计
        headers = {"ID", "时间", "图片目录", "总图片数", "已标注", "标注框数", "标签列表"};
        q = m_db->queryAnnotationStats();
        break;
    }
    case 2: { // 量化记录
        headers = {"ID", "时间", "源模型", "量化类型", "输出模型", "原始大小(MB)",
                   "量化大小(MB)", "状态"};
        q = m_db->queryQuantRecords();
        break;
    }
    case 3: { // 推理结果
        headers = {"ID", "时间", "模型", "图片", "检测数", "最高置信度",
                   "推理耗时(ms)"};
        q = m_db->queryInferResults();
        break;
    }
    default:
        return;
    }

    dbFillTableWidget(q, headers);
}

void MainWindow::dbFillTableWidget(QSqlQuery &q, const QStringList &headers)
{
    QTableWidget *tw = ui->tableWidget_dbResult;
    tw->clear();
    tw->setColumnCount(headers.size());
    tw->setHorizontalHeaderLabels(headers);
    tw->horizontalHeader()->setStretchLastSection(true);
    tw->setRowCount(0);

    int row = 0;
    while (q.next()) {
        tw->insertRow(row);
        QSqlRecord rec = q.record();
        // 填写前 headers.size() 列（跳过 notes / detections_json 等冗长字段）
        for (int col = 0; col < qMin(headers.size(), rec.count()); ++col) {
            QString val = q.value(col).toString();
            QTableWidgetItem *item = new QTableWidgetItem(val);
            item->setTextAlignment(Qt::AlignCenter);
            // 状态列上色
            if (headers[col] == "状态") {
                if (val == "completed")
                    item->setForeground(QColor("#4CAF50"));
                else if (val == "failed" || val == "crashed")
                    item->setForeground(QColor("#f44336"));
                else
                    item->setForeground(QColor("#FF9800"));
            }
            tw->setItem(row, col, item);
        }
        ++row;
    }

    tw->resizeColumnsToContents();
    tw->horizontalHeader()->setStretchLastSection(true);
}


// ===== Tab4 槽函数 =====

void MainWindow::on_btn_dbRefresh_clicked()
{
    dbRefreshStats();
    dbLoadTable(ui->combo_dbTable->currentIndex());
    ui->statusbar->showMessage("数据已刷新", 2000);
}

void MainWindow::on_combo_dbTable_currentIndexChanged(int index)
{
    dbLoadTable(index);
}

void MainWindow::on_btn_dbExport_clicked()
{
    if (!m_db->isOpen()) {
        QMessageBox::warning(this, "错误", "数据库未就绪！");
        return;
    }

    QString fileName = QFileDialog::getSaveFileName(
        this, "导出 CSV 文件",
        "D:/YC/puddle_export.csv",
        "CSV Files (*.csv);;All Files (*)"
    );
    if (fileName.isEmpty()) return;

    QFile file(fileName);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::warning(this, "错误", "无法创建文件：" + fileName);
        return;
    }

    QTextStream out(&file);
    out.setEncoding(QStringConverter::Utf8);
    // 写 BOM（让 Excel 正确识别 UTF-8）
    out << "\xEF\xBB\xBF";

    QTableWidget *tw = ui->tableWidget_dbResult;
    // 表头
    QStringList headerRow;
    for (int c = 0; c < tw->columnCount(); ++c)
        headerRow << tw->horizontalHeaderItem(c)->text();
    out << headerRow.join(",") << "\n";

    // 数据行
    for (int r = 0; r < tw->rowCount(); ++r) {
        QStringList row;
        for (int c = 0; c < tw->columnCount(); ++c) {
            QTableWidgetItem *item = tw->item(r, c);
            QString val = item ? item->text() : "";
            // 含逗号/引号的字段加双引号
            if (val.contains(',') || val.contains('"'))
                val = "\"" + val.replace("\"", "\"\"") + "\"";
            row << val;
        }
        out << row.join(",") << "\n";
    }

    file.close();
    QMessageBox::information(this, "导出成功",
        QString("已导出 %1 行数据到：\n%2").arg(tw->rowCount()).arg(fileName));
}

void MainWindow::on_btn_dbSaveAnnot_clicked()
{
    if (!m_db->isOpen()) {
        QMessageBox::warning(this, "错误", "数据库未就绪！");
        return;
    }

    if (m_annotCurrentDir.isEmpty()) {
        QMessageBox::information(this, "提示",
            "请先在「图像标注」页签中打开一个图片文件夹。");
        return;
    }

    // 统计标注信息
    int totalImages   = m_annotImageFiles.size();
    int labeledImages = 0;
    int totalBoxes    = 0;

    for (const QString &imgPath : m_annotImageFiles) {
        QString labelPath = annotImagePathToLabelPath(imgPath);
        QFile f(labelPath);
        if (!f.exists()) continue;
        if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) continue;
        QTextStream ts(&f);
        int lineCount = 0;
        while (!ts.atEnd()) {
            QString line = ts.readLine().trimmed();
            if (!line.isEmpty()) ++lineCount;
        }
        f.close();
        if (lineCount > 0) {
            ++labeledImages;
            totalBoxes += lineCount;
        }
    }

    // 标签列表
    QJsonArray labelArr;
    for (int i = 0; i < m_labelModel->rowCount(); ++i)
        labelArr.append(m_labelModel->item(i, 0)->text());
    QString labelJson = QString::fromUtf8(
        QJsonDocument(labelArr).toJson(QJsonDocument::Compact)
    );

    qDebug() << "[DB] 准备写入标注统计："
             << "dir=" << m_annotCurrentDir
             << "total=" << totalImages
             << "labeled=" << labeledImages
             << "boxes=" << totalBoxes;

    qint64 rid = m_db->insertAnnotationStat(
        m_annotCurrentDir,
        totalImages, labeledImages, totalBoxes,
        labelJson
    );

    qDebug() << "[DB] insertAnnotationStat 返回 rid=" << rid;

    if (rid > 0) {
        QMessageBox::information(this, "已保存",
            QString("标注统计已保存到数据库（ID=%1）\n"
                    "目录：%2\n"
                    "图片总数：%3  已标注：%4  标注框总数：%5")
            .arg(rid).arg(m_annotCurrentDir)
            .arg(totalImages).arg(labeledImages).arg(totalBoxes));
        dbRefreshStats();
        dbLoadTable(1); // 切换到标注统计表
        ui->combo_dbTable->setCurrentIndex(1);
    } else {
        QMessageBox::warning(this, "保存失败",
            "写入数据库失败：" + m_db->lastError());
    }
}
