#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QProcess>
#include <QFileDialog>
#include <QRegularExpression>
#include <QTextCursor>
#include <QScrollBar>
#include <QDateTime>
#include <QStandardItemModel>
#include <QTableView>
#include <QTableWidget>
#include <QListWidget>
#include <QSqlQuery>
#include "annotationview.h"
#include "db_manager.h"
#include "inference_cpp.h"
#include "losscurveview.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

private slots:
    // 兼容旧版 pushButton，防止编译报错
    void on_pushButton_clicked();

    // ===== 训练相关 =====
    void on_btn_selectDataset_clicked();
    void on_btn_selectPython_clicked();
    void on_btn_train_clicked();
    void on_btn_stop_clicked();
    void on_btn_clearLog_clicked();
    void onProcessReadyRead();
    void onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void onProcessError(QProcess::ProcessError error);

    // ===== 标注相关 =====
    void on_btn_annotOpenFolder_clicked();
    void on_btn_annotPrev_clicked();
    void on_btn_annotNext_clicked();
    void on_btn_annotAddLabel_clicked();
    void on_btn_annotDelLabel_clicked();
    void on_btn_annotSave_clicked();
    void on_btn_annotSelectModel_clicked();
    void on_btn_annotAutoLabel_clicked();
    void on_listWidget_images_itemClicked(QListWidgetItem *item);
    void on_annotBoxesChanged(const QVector<BoundingBox> &boxes);
    void on_annotAutoReadyRead();
    void on_annotAutoFinished(int exitCode, QProcess::ExitStatus status);
    void on_annotationLabelChanged(const QString &label);

    // ===== 标注信息列表 + 改类别 =====
    void on_annotSelectionChanged();                    // 选中框变化时更新右侧信息列表
    void on_btn_annotChangeLabel_clicked();             // 把选中框改成 tableView_labels 当前选中的标签
    void on_tableView_labels_clicked(const QModelIndex &index); // 标签表点击
    void updateAnnotInfoList();                         // 刷新 listWidget_annotInfo

    // ===== 量化相关 =====
    void on_btn_quantSelectModel_clicked();
    void on_btn_quantSelectCalib_clicked();
    void on_btn_quantSelectOutput_clicked();
    void on_btn_quantStart_clicked();
    void on_btn_quantStop_clicked();
    void on_btn_quantClearLog_clicked();
    void onQuantReadyRead();
    void onQuantFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void onQuantError(QProcess::ProcessError error);

    // ===== 量化后推理相关 =====
    void on_btn_inferSelectModel_clicked();
    void on_btn_inferSelectImage_clicked();
    void on_btn_inferRun_clicked();
    void onInferReadyRead();
    void onInferFinished(int exitCode, QProcess::ExitStatus exitStatus);

    // ===== 数据库与可视化相关 =====
    void on_btn_dbRefresh_clicked();
    void on_btn_dbExport_clicked();
    void on_btn_dbSaveAnnot_clicked();
    void on_combo_dbTable_currentIndexChanged(int index);

private:
    Ui::MainWindow *ui;
    QProcess       *m_process;
    int             m_totalEpochs;
    int             m_currentEpoch;

    // 训练结果指标
    double          m_lastMap50     = -1.0;
    double          m_lastPrecision = -1.0;
    double          m_lastRecall    = -1.0;

    // 标注相关
    AnnotationView       *m_annotView;
    QStandardItemModel   *m_labelModel;
    QProcess             *m_autoProcess;
    QStringList           m_annotImageFiles;
    int                   m_annotCurrentIndex = -1;
    QString               m_annotCurrentDir;
    QString               m_yoloModelPath;
    QString               m_annotDirtyFlag;

    // 量化相关
    QProcess             *m_quantProcess;
    QString               m_quantOutputPath;   // 量化后模型路径（由量化脚本输出）

    // 推理相关
    QProcess             *m_inferProcess;
    QString               m_inferModelPath;    // 当前选定的量化 ONNX 模型
    QString               m_inferImagePath;    // 当前选定的推理图片
    CppInference           m_cppInfer;         // C++ 推理引擎
    void runCppInference();                    // 执行 C++ 推理

    // 数据库
    DatabaseManager      *m_db;

    // 训练曲线历史（按 epoch 顺序）
    QVector<QPointF> m_lossClsHistory;
    QVector<QPointF> m_lossDflHistory;
    QVector<QPointF> m_mapHistory;       // mAP50 逐轮历史
    QVector<QPointF> m_precHistory;      // Precision 逐轮历史
    QVector<QPointF> m_recHistory;       // Recall 逐轮历史
    int m_lastEpochInCurve = 0;
    QString m_trainScriptDir;  // train.py 所在目录，用于定位 runs/ 输出

    void appendLog(const QString &text);
    void setTrainingState(bool training);
    void tryParseEpoch(const QString &line);
    void tryParseMetrics(const QString &line);
    void refreshLossCurve();
    QString findBestPt(const QString &outputDir);

    // 标注工具函数
    void annotLoadImage(const QString &path);
    void annotSaveAnnotations();
    void annotSyncLabels();
    void annotAddDefaultLabels();
    QString annotImagePathToLabelPath(const QString &imgPath) const;

    // 量化工具函数
    void appendQuantLog(const QString &text);
    void setQuantState(bool running);
    void loadInferResult(const QString &jsonPath);

    // 推理结果显示
    void showInferImage(const QString &imagePath, const QJsonArray &detections);

    // 数据库工具函数
    void dbInit();
    void dbRefreshStats();
    void dbLoadTable(int tableIndex);
    void dbFillTableWidget(QSqlQuery &q, const QStringList &headers);
};

#endif // MAINWINDOW_H

