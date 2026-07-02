#include "annotationwindow.h"
#include "ui_annotationwindow.h"
#include <QMenuBar>
#include <QToolBar>
#include <QStatusBar>
#include <QDockWidget>
#include <QListView>
#include <QTableView>
#include <QFileDialog>
#include <QInputDialog>
#include <QModelIndex>
#include <QSortFilterProxyModel>
#include <QMessageBox>
#include <QVBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QPainter>
#include <QFileInfo>
#include <QDir>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QRegularExpression>
#include <QHeaderView>

// ============================================================
// 工具：归一化坐标转 YOLO txt 格式（中心x 中心y 宽 高，都是归一化）
// ============================================================
static QString boxToYoloTxt(const BoundingBox &box, int imgW, int imgH)
{
    double cx = box.rect.center().x();
    double cy = box.rect.center().y();
    double w  = box.rect.width();
    double h  = box.rect.height();
    return QString("%1 %2 %3 %4\n").arg(cx, 0, 'g', 6)
                                   .arg(cy, 0, 'g', 6)
                                   .arg(w,  0, 'g', 6)
                                   .arg(h,  0, 'g', 6);
}

// ============================================================
// 构造
// ============================================================
AnnotationWindow::AnnotationWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::AnnotationWindow)
    , m_canvas(new AnnotationView(this))
    , m_fileModel(new QFileSystemModel(this))
    , m_labelModel(new QStandardItemModel(this))
    , m_autoProcess(new QProcess(this))
{
    ui->setupUi(this);
    setWindowTitle("水坑检测 — 图像标注工具");
    resize(1400, 900);

    // ---- 菜单栏 ----
    QMenu *fileMenu = menuBar()->addMenu("文件(&F)");
    QAction *actOpenImg = new QAction("打开图片...", this);
    actOpenImg->setShortcut(QKeySequence("Ctrl+O"));
    connect(actOpenImg, &QAction::triggered, this, &AnnotationWindow::onOpenImage);
    fileMenu->addAction(actOpenImg);
    
    QAction *actOpenDir = new QAction("打开文件夹...", this);
    actOpenDir->setShortcut(QKeySequence("Ctrl+Shift+O"));
    connect(actOpenDir, &QAction::triggered, this, &AnnotationWindow::onOpenFolder);
    fileMenu->addAction(actOpenDir);
    
    fileMenu->addSeparator();
    
    QAction *actSave = new QAction("保存", this);
    actSave->setShortcut(QKeySequence("Ctrl+S"));
    connect(actSave, &QAction::triggered, this, &AnnotationWindow::onSave);
    fileMenu->addAction(actSave);
    
    QAction *actSaveAs = new QAction("另存为...", this);
    actSaveAs->setShortcut(QKeySequence("Ctrl+Shift+S"));
    connect(actSaveAs, &QAction::triggered, this, &AnnotationWindow::onSaveAs);
    fileMenu->addAction(actSaveAs);
    
    fileMenu->addSeparator();
    fileMenu->addAction("退出", this, &QWidget::close);

    QMenu *editMenu = menuBar()->addMenu("编辑(&E)");
    QAction *actUndo = new QAction("撤销", this);
    actUndo->setShortcut(QKeySequence("Ctrl+Z"));
    connect(actUndo, &QAction::triggered, m_canvas, &AnnotationView::undo);
    editMenu->addAction(actUndo);
    
    QAction *actRedo = new QAction("重做", this);
    actRedo->setShortcut(QKeySequence("Ctrl+Y"));
    connect(actRedo, &QAction::triggered, m_canvas, &AnnotationView::redo);
    editMenu->addAction(actRedo);

    QMenu *viewMenu = menuBar()->addMenu("视图(&V)");
    QAction *actZoomIn = new QAction("放大", this);
    actZoomIn->setShortcut(QKeySequence("Ctrl++"));
    connect(actZoomIn, &QAction::triggered, this, &AnnotationWindow::onZoomIn);
    viewMenu->addAction(actZoomIn);
    
    QAction *actZoomOut = new QAction("缩小", this);
    actZoomOut->setShortcut(QKeySequence("Ctrl+-"));
    connect(actZoomOut, &QAction::triggered, this, &AnnotationWindow::onZoomOut);
    viewMenu->addAction(actZoomOut);
    
    QAction *actZoomFit = new QAction("适应窗口", this);
    actZoomFit->setShortcut(QKeySequence("Ctrl+0"));
    connect(actZoomFit, &QAction::triggered, this, &AnnotationWindow::onZoomFit);
    viewMenu->addAction(actZoomFit);
    
    QAction *actZoom100 = new QAction("100%", this);
    connect(actZoom100, &QAction::triggered, this, &AnnotationWindow::onZoom100);
    viewMenu->addAction(actZoom100);

    QMenu *labelMenu = menuBar()->addMenu("标签(&L)");
    QAction *actAddLabel = new QAction("添加标签...", this);
    actAddLabel->setShortcut(QKeySequence("Ctrl+T"));
    connect(actAddLabel, &QAction::triggered, this, &AnnotationWindow::onAddLabel);
    labelMenu->addAction(actAddLabel);
    
    QAction *actDelLabel = new QAction("删除选中标签", this);
    connect(actDelLabel, &QAction::triggered, this, &AnnotationWindow::onDeleteLabel);
    labelMenu->addAction(actDelLabel);
    
    labelMenu->addSeparator();
    
    QAction *actAutoLabel = new QAction("自动标注（YOLO）...", this);
    connect(actAutoLabel, &QAction::triggered, this, &AnnotationWindow::onAutoLabel);
    labelMenu->addAction(actAutoLabel);
    
    QAction *actSelectModel = new QAction("选择模型文件...", this);
    connect(actSelectModel, &QAction::triggered, this, &AnnotationWindow::onSelectModel);
    labelMenu->addAction(actSelectModel);

    // ---- 工具栏 ----
    QToolBar *toolBar = addToolBar("工具");
    toolBar->setIconSize(QSize(20, 20));

    QPushButton *btnOpenImg = new QPushButton("📷 打开图片");
    QPushButton *btnOpenDir = new QPushButton("📁 打开文件夹");
    QPushButton *btnSave    = new QPushButton("💾 保存");
    QPushButton *btnAuto   = new QPushButton("🤖 自动标注");
    QPushButton *btnFit    = new QPushButton("🔲 适应");
    QPushButton *btnZoomIn = new QPushButton("🔍+");
    QPushButton *btnZoomOut= new QPushButton("🔍-");
    QPushButton *btnDraw   = new QPushButton("✏️ 绘制");
    btnDraw->setCheckable(true);
    QPushButton *btnSelect = new QPushButton("🖱 选择");
    btnSelect->setCheckable(true);
    btnSelect->setChecked(true);  // 默认选择模式

    connect(btnOpenImg,  &QPushButton::clicked, this, &AnnotationWindow::onOpenImage);
    connect(btnOpenDir,  &QPushButton::clicked, this, &AnnotationWindow::onOpenFolder);
    connect(btnSave,     &QPushButton::clicked, this, &AnnotationWindow::onSave);
    connect(btnAuto,     &QPushButton::clicked, this, &AnnotationWindow::onAutoLabel);
    connect(btnFit,      &QPushButton::clicked, this, &AnnotationWindow::onZoomFit);
    connect(btnZoomIn,   &QPushButton::clicked, this, &AnnotationWindow::onZoomIn);
    connect(btnZoomOut,  &QPushButton::clicked, this, &AnnotationWindow::onZoomOut);
    connect(btnDraw,     &QPushButton::clicked, this, [this](){
        m_canvas->setTool(AnnotationView::TOOL_DRAW);
        btnDraw->setChecked(true);
        btnSelect->setChecked(false);
    });
    connect(btnSelect,   &QPushButton::clicked, this, [this](){
        m_canvas->setTool(AnnotationView::TOOL_SELECT);
        btnSelect->setChecked(true);
        btnDraw->setChecked(false);
    });

    for (auto *btn : {btnOpenImg, btnOpenDir, btnSave, btnAuto,
                      btnFit, btnZoomIn, btnZoomOut, btnDraw, btnSelect})
        toolBar->addWidget(btn);

    // ---- 中央：标注画布（已在初始化列表创建）----
    setCentralWidget(m_canvas);

    // ---- 左侧停靠：图片列表 ----
    QDockWidget *dockImg = new QDockWidget("图片列表", this);
    dockImg->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable);
    dockImg->setMinimumWidth(220);
    QVBoxLayout *imgListLayout = new QVBoxLayout;
    imgListLayout->setContentsMargins(4,4,4,4);

    // 导航按钮
    QHBoxLayout *navLayout = new QHBoxLayout;
    QPushButton *btnPrev = new QPushButton("◀ 上一张");
    QPushButton *btnNext = new QPushButton("下一张 ▶");
    btnPrev->setMaximumWidth(80);
    btnNext->setMaximumWidth(80);
    connect(btnPrev, &QPushButton::clicked, this, &AnnotationWindow::onPrevImage);
    connect(btnNext, &QPushButton::clicked, this, &AnnotationWindow::onNextImage);
    navLayout->addWidget(btnPrev);
    navLayout->addWidget(btnNext);

    QListView *imgListView = new QListView;
    imgListView->setIconSize(QSize(80, 80));
    imgListView->setViewMode(QListView::IconMode);
    imgListView->setGridSize(QSize(100, 100));
    m_fileModel->setNameFilters({"*.jpg","*.jpeg","*.png","*.bmp","*.webp"});
    m_fileModel->setNameFilterDisables(false);
    QSortFilterProxyModel *proxy = new QSortFilterProxyModel(this);
    proxy->setSourceModel(m_fileModel);
    proxy->setFilterRegularExpression("\\.(jpg|jpeg|png|bmp|webp)$");
    imgListView->setModel(proxy);

    connect(imgListView, &QListView::clicked, this, [=](const QModelIndex &idx){
        QString path = m_fileModel->filePath(proxy->mapToSource(idx));
        if (!path.isEmpty()) loadImage(path);
    });

    imgListLayout->addLayout(navLayout);
    imgListLayout->addWidget(imgListView);

    QWidget *imgListWidget = new QWidget;
    imgListWidget->setLayout(imgListLayout);
    dockImg->setWidget(imgListWidget);
    addDockWidget(Qt::LeftDockWidgetArea, dockImg);

    // ---- 右侧停靠：标签管理 ----
    QDockWidget *dockLabel = new QDockWidget("标签管理", this);
    dockLabel->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable);
    dockLabel->setMinimumWidth(200);
    QVBoxLayout *labelLayout = new QVBoxLayout;

    QPushButton *btnAddLabel    = new QPushButton("➕ 添加标签");
    QPushButton *btnDelLabel    = new QPushButton("🗑 删除标签");
    connect(btnAddLabel, &QPushButton::clicked, this, &AnnotationWindow::onAddLabel);
    connect(btnDelLabel, &QPushButton::clicked, this, &AnnotationWindow::onDeleteLabel);

    m_labelModel->setHorizontalHeaderLabels({"标签", "颜色", "可见"});
    m_labelTable = new QTableView;
    m_labelTable->setModel(m_labelModel);
    m_labelTable->setColumnWidth(0, 100);
    m_labelTable->setColumnWidth(1, 50);
    m_labelTable->setColumnWidth(2, 40);
    m_labelTable->horizontalHeader()->setStretchLastSection(true);
    connect(m_labelModel, &QStandardItemModel::itemChanged,
            this, &AnnotationWindow::onLabelItemChanged);

    labelLayout->addWidget(btnAddLabel);
    labelLayout->addWidget(btnDelLabel);
    labelLayout->addWidget(m_labelTable);

    QWidget *labelWidget = new QWidget;
    labelWidget->setLayout(labelLayout);
    dockLabel->setWidget(labelWidget);
    addDockWidget(Qt::RightDockWidgetArea, dockLabel);

    // ---- 底部停靠：标注信息 ----
    QDockWidget *dockInfo = new QDockWidget("标注信息", this);
    QVBoxLayout *infoLayout = new QVBoxLayout;
    QLabel *lblInfo = new QLabel("未加载图片");
    lblInfo->setStyleSheet("color:#888; font-size:12px;");
    m_lblInfo = lblInfo;
    infoLayout->addWidget(lblInfo);
    QWidget *infoWidget = new QWidget;
    infoWidget->setLayout(infoLayout);
    dockInfo->setWidget(infoWidget);
    addDockWidget(Qt::BottomDockWidgetArea, dockInfo);

    // ---- 信号绑定 ----
    connect(m_canvas, &AnnotationView::boxesChanged,
            this, &AnnotationWindow::onBoxesChanged);
    connect(m_canvas, &AnnotationView::currentLabelChanged,
            this, [this](const QString &){ /* 标签切换处理 */ });

    connect(m_autoProcess, &QProcess::readyReadStandardOutput,
            this, &AnnotationWindow::onAutoLabelReadyRead);
    connect(m_autoProcess, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &AnnotationWindow::onAutoLabelFinished);

    // ---- 初始化默认标签 ----
    addDefaultLabels();

    statusBar()->showMessage("就绪");
}

AnnotationWindow::~AnnotationWindow()
{
    delete ui;
}

void AnnotationWindow::addDefaultLabels()
{
    // 添加默认标签（与数据集匹配）
    static const QStringList DEFAULT_LABELS = {
        "puddle"
    };
    for (const QString &name : DEFAULT_LABELS) {
        QList<QStandardItem*> row;
        row << new QStandardItem(name);
        row << new QStandardItem(); // 颜色（用默认）
        row << new QStandardItem("✓");
        m_labelModel->appendRow(row);
    }
    syncLabelsToCanvas();
    if (!DEFAULT_LABELS.isEmpty())
        m_canvas->setCurrentLabel(DEFAULT_LABELS.first());
}

// ============================================================
// 打开图片
// ============================================================
void AnnotationWindow::onOpenImage()
{
    if (!askSave()) return;
    QString path = QFileDialog::getOpenFileName(
        this, "选择图片", m_currentDir,
        "Images (*.jpg *.jpeg *.png *.bmp *.webp)");
    if (!path.isEmpty()) loadImage(path);
}

void AnnotationWindow::onOpenFolder()
{
    if (!askSave()) return;
    QString dir = QFileDialog::getExistingDirectory(
        this, "选择图片文件夹", m_currentDir);
    if (dir.isEmpty()) return;

    m_currentDir = dir;
    m_fileModel->setRootPath(dir);
    
    // 填充 m_imageFiles 列表
    m_imageFiles.clear();
    QDir d(dir);
    const QStringList filters = {"*.jpg","*.jpeg","*.png","*.bmp","*.webp"};
    for (const QString &f : d.entryList(filters, QDir::Files))
        m_imageFiles.append(dir + "/" + f);
    
    m_currentIndex = -1;
    statusBar()->showMessage(QString("已打开文件夹，共 %1 张图片").arg(m_imageFiles.size()), 3000);
}

// ============================================================
// 加载图片
// ============================================================
void AnnotationWindow::loadImage(const QString &path)
{
    if (!askSave()) return;

    QImage img(path);
    if (img.isNull()) {
        QMessageBox::warning(this, "错误", "无法加载图片：" + path);
        return;
    }

    m_currentImagePath = path;
    m_canvas->setImage(img);

    // 加载该图的标注
    QString labelPath = imagePathToLabelPath(path);
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
                bool ok1,ok2,ok3,ok4;
                double cx = parts[0].toDouble(&ok1);
                double cy = parts[1].toDouble(&ok2);
                double w  = parts[2].toDouble(&ok3);
                double h  = parts[3].toDouble(&ok4);
                int classId = parts[4].toInt();
                if (!ok1 || !ok2 || !ok3 || !ok4) continue;

                BoundingBox box;
                box.id = id++;
                box.rect = QRectF(cx - w/2, cy - h/2, w, h);
                // 找标签名
                if (classId >= 0 && classId < m_labels.size())
                    box.label = m_labels[classId].name;
                else
                    box.label = "unknown";
                boxes.append(box);
            }
            f.close();
        }
    }

    m_canvas->setBoxes(boxes);
    m_dirty = false;
    updateStatus();
    updateImageNavigationButtons();
}

// ============================================================
// 导航
// ============================================================
void AnnotationWindow::onPrevImage()
{
    if (m_currentIndex > 0) {
        --m_currentIndex;
        loadImage(m_imageFiles[m_currentIndex]);
    }
}

void AnnotationWindow::onNextImage()
{
    if (m_currentIndex < m_imageFiles.size() - 1) {
        ++m_currentIndex;
        loadImage(m_imageFiles[m_currentIndex]);
    }
}

void AnnotationWindow::onImageListClicked(const QModelIndex &index)
{
    if (!index.isValid()) return;
    QString fileName = index.data().toString();
    QString fullPath = m_currentDir + "/" + fileName;
    for (int i = 0; i < m_imageFiles.size(); ++i) {
        if (m_imageFiles[i] == fullPath) {
            m_currentIndex = i;
            loadImage(fullPath);
            break;
        }
    }
}

void AnnotationWindow::onLabelColorChange()
{
    syncLabelsToCanvas();
}

void AnnotationWindow::onSelectTool()
{
    if (m_canvas) m_canvas->setTool(AnnotationView::TOOL_SELECT);
}

void AnnotationWindow::onDrawTool()
{
    if (m_canvas) m_canvas->setTool(AnnotationView::TOOL_DRAW);
}

void AnnotationWindow::updateImageNavigationButtons()
{
    // 更新列表中当前项的选中状态
}

// ============================================================
// 保存
// ============================================================
void AnnotationWindow::onSave()
{
    saveAnnotations();
    m_dirty = false;
    statusBar()->showMessage("✅ 已保存", 2000);
}

void AnnotationWindow::onSaveAs()
{
    QString path = QFileDialog::getSaveFileName(
        this, "导出标注", m_currentImagePath,
        "YOLO Text (*.txt)");
    if (path.isEmpty()) return;
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::warning(this, "错误", "无法写入文件：" + path);
        return;
    }
    QTextStream ts(&f);
    ts.setEncoding(QStringConverter::Utf8);
    QVector<BoundingBox> boxes = m_canvas->getBoxes();
    if (!m_canvas->getImage().isNull()) {
        int w = m_canvas->getImage().width();
        int h = m_canvas->getImage().height();
        for (const auto &box : boxes) {
            ts << boxToYoloTxt(box, w, h);
        }
    }
    f.close();
    statusBar()->showMessage("已导出到：" + path, 3000);
}

void AnnotationWindow::loadAnnotations()
{
    // 标注已在 loadImage() 中同步加载
}

void AnnotationWindow::saveAnnotations()
{
    if (m_currentImagePath.isEmpty()) return;
    QVector<BoundingBox> boxes = m_canvas->getBoxes();
    if (boxes.isEmpty()) {
        // 如果没有标注，删除 label 文件
        QString lp = imagePathToLabelPath(m_currentImagePath);
        if (QFile::exists(lp)) QFile::remove(lp);
        return;
    }

    QImage img = m_canvas->getImage();
    if (img.isNull()) return;
    int w = img.width(), h = img.height();

    QString labelPath = imagePathToLabelPath(m_currentImagePath);
    QFile f(labelPath);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) return;
    QTextStream ts(&f);
    ts.setEncoding(QStringConverter::Utf8);

    // 建立标签名->ID映射
    QMap<QString, int> labelToId;
    for (int i = 0; i < m_labels.size(); ++i)
        labelToId[m_labels[i].name] = i;

    for (const auto &box : boxes) {
        int classId = labelToId.value(box.label, 0);
        ts << QString::number(classId) << " "
           << boxToYoloTxt(box, w, h).trimmed() << "\n";
    }
    f.close();
}

QString AnnotationWindow::imagePathToLabelPath(const QString &imgPath) const
{
    QFileInfo fi(imgPath);
    return fi.absolutePath() + "/" + fi.baseName() + ".txt";
}

bool AnnotationWindow::askSave()
{
    if (!m_dirty) return true;
    int ret = QMessageBox::question(this, "保存更改",
        "当前标注尚未保存，是否保存？",
        QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel);
    if (ret == QMessageBox::Save) {
        saveAnnotations();
        m_dirty = false;
        return true;
    }
    return ret == QMessageBox::Discard;
}

// ============================================================
// 标注框变化
// ============================================================
void AnnotationWindow::onBoxesChanged(const QVector<BoundingBox> &)
{
    m_dirty = true;
    updateStatus();
}

// ============================================================
// 标签管理
// ============================================================
void AnnotationWindow::onAddLabel()
{
    bool ok;
    QString name = QInputDialog::getText(this, "添加标签", "标签名称：",
                                          QLineEdit::Normal, "", &ok);
    if (!ok || name.isEmpty()) return;

    // 检查重复
    for (const auto &l : m_labels) {
        if (l.name == name) {
            QMessageBox::warning(this, "错误", "标签已存在：" + name);
            return;
        }
    }

    QList<QStandardItem*> row;
    row << new QStandardItem(name);
    row << new QStandardItem(); // 默认颜色
    row << new QStandardItem("✓");
    m_labelModel->appendRow(row);
    syncLabelsToCanvas();
}

void AnnotationWindow::onDeleteLabel()
{
    QModelIndex idx;
    if (m_labelTable) idx = m_labelTable->currentIndex();
    if (!idx.isValid()) {
        QMessageBox::information(this, "提示", "请先在标签列表中选中要删除的标签");
        return;
    }
    int row = idx.row();
    QString name = m_labelModel->item(row, 0)->text();
    m_labelModel->removeRow(row);
    syncLabelsToCanvas();
    statusBar()->showMessage("已删除标签：" + name, 2000);
}

void AnnotationWindow::onLabelItemChanged(QStandardItem *item)
{
    Q_UNUSED(item);
    syncLabelsToCanvas();
}

// ============================================================
// 标签颜色池
// ============================================================
static QColor nextColor(int idx) {
    const QList<QColor> pool = {
        QColor("#FF6B6B"), QColor("#4ECDC4"), QColor("#45B7D1"),
        QColor("#96CEB4"), QColor("#FFEAA7"), QColor("#DDA0DD"),
        QColor("#98D8C8"), QColor("#F7DC6F"), QColor("#BB8FCE"),
        QColor("#85C1E9"), QColor("#F8B500"), QColor("#00CED1"),
    };
    return pool[idx % pool.size()];
}

// ============================================================
// 标签管理
// ============================================================
void AnnotationWindow::syncLabelsToCanvas()
{
    QVector<LabelInfo> labels;
    for (int i = 0; i < m_labelModel->rowCount(); ++i) {
        LabelInfo li;
        li.name = m_labelModel->item(i, 0)->text();
        li.color = nextColor(i);
        li.visible = (m_labelModel->item(i, 2)->text() == "✓");
        labels.append(li);
    }
    m_labels = labels;
    m_canvas->setLabels(labels);
}

// ============================================================
// 自动标注（YOLO 推理）
// ============================================================
void AnnotationWindow::onAutoLabel()
{
    if (m_currentImagePath.isEmpty()) {
        QMessageBox::information(this, "提示", "请先打开一张图片");
        return;
    }

    if (m_yoloModelPath.isEmpty()) {
        QMessageBox::information(this, "提示",
            "请先点击「选择模型文件」选择 best.pt 模型路径");
        return;
    }

    if (!QFile::exists(m_yoloModelPath)) {
        QMessageBox::warning(this, "错误", "模型文件不存在：" + m_yoloModelPath);
        return;
    }

    if (m_autoProcess->state() != QProcess::NotRunning) {
        QMessageBox::information(this, "提示", "自动标注正在进行中...");
        return;
    }

    // 写一个临时推理脚本
    QString scriptPath = QDir::temp().filePath("predict_annotation.py");
    QFile f(scriptPath);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) return;
    QTextStream ts(&f);
    ts << "# auto predict script\n";
    ts << "import sys, os, json\n";
    ts << "sys.stdout.reconfigure(encoding='utf-8')\n";
    ts << "from ultralytics import YOLO\n";
    ts << "model = YOLO('" << m_yoloModelPath << "')\n";
    ts << "img = r'" << m_currentImagePath << "'\n";
    ts << "results = model.predict(img, verbose=False, conf=0.25)\n";
    ts << "boxes = results[0].boxes\n";
    ts << "img_h, img_w = results[0].orig_shape\n";
    ts << "for box in boxes:\n";
    ts << "    x1,y1,x2,y2 = box.xywhn[0].tolist()\n";
    ts << "    cls_id = int(box.cls[0])\n";
    ts << "    print(f'BOX:{x1:.6f},{y1:.6f},{x2:.6f},{y2:.6f},{cls_id}', flush=True)\n";
    ts << "print('DONE', flush=True)\n";
    f.close();

    m_autoProcess->setProcessChannelMode(QProcess::MergedChannels);
    m_autoProcess->start("D:/Python/dog/.venv/Scripts/python.exe",
                          QStringList() << scriptPath);
    statusBar()->showMessage("🤖 自动标注中...");
}

void AnnotationWindow::onAutoLabelReadyRead()
{
    QByteArray data = m_autoProcess->readAllStandardOutput();
    QString text = QString::fromUtf8(data);
    // 解析 BOX:cx,cy,w,h,cls
    static QRegularExpression re("BOX:([\\d.]+),([\\d.]+),([\\d.]+),([\\d.]+),(\\d+)");
    QRegularExpressionMatchIterator it = re.globalMatch(text);
    QVector<BoundingBox> boxes;
    int id = 0;
    while (it.hasNext()) {
        QRegularExpressionMatch m = it.next();
        BoundingBox b;
        b.id = id++;
        b.rect = QRectF(
            m.captured(1).toDouble() - m.captured(3).toDouble()/2,
            m.captured(2).toDouble() - m.captured(4).toDouble()/2,
            m.captured(3).toDouble(),
            m.captured(4).toDouble()
        );
        int clsId = m.captured(5).toInt();
        if (clsId >= 0 && clsId < m_labels.size())
            b.label = m_labels[clsId].name;
        else
            b.label = "unknown";
        boxes.append(b);
    }
    if (!boxes.isEmpty() && text.contains("DONE")) {
        m_canvas->setBoxes(boxes);
        statusBar()->showMessage(
            QString("✅ 自动标注完成，找到 %1 个目标").arg(boxes.size()), 5000);
    }
}

void AnnotationWindow::onAutoLabelFinished(int exitCode, QProcess::ExitStatus)
{
    if (exitCode != 0)
        statusBar()->showMessage("❌ 自动标注失败", 3000);
}

void AnnotationWindow::onSelectModel()
{
    QString path = QFileDialog::getOpenFileName(
        this, "选择 YOLO 模型", "D:/YC/software/QT/PuddleDetection/runs/train",
        "Model Files (*.pt);;All Files (*)");
    if (!path.isEmpty()) {
        m_yoloModelPath = path;
        statusBar()->showMessage("模型已选择：" + path, 3000);
    }
}

// ============================================================
// 状态更新
// ============================================================
void AnnotationWindow::updateStatus()
{
    if (m_currentImagePath.isEmpty()) {
        m_lblInfo->setText("未加载图片");
        return;
    }
    QFileInfo fi(m_currentImagePath);
    QVector<BoundingBox> boxes = m_canvas->getBoxes();
    QString text = QString("文件: %1 (%2×%3)  |  标注数: %4  %5")
        .arg(fi.fileName())
        .arg(m_canvas->getImage().width())
        .arg(m_canvas->getImage().height())
        .arg(boxes.size())
        .arg(m_dirty ? "⚠️ 未保存" : "");
    m_lblInfo->setText(text);
}
