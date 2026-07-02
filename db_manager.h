#ifndef DB_MANAGER_H
#define DB_MANAGER_H

#include <QObject>
#include <QSqlDatabase>
#include <QSqlQuery>
#include <QSqlError>
#include <QDateTime>
#include <QVariant>
#include <QDebug>

// ============================================================
//  DatabaseManager — 封装 SQLite 数据库操作
//  数据库文件：应用目录下的 puddle_detection.db
//
//  表结构：
//  1. train_records    — 训练记录
//  2. annotation_stats — 标注统计
//  3. quant_records    — 量化记录
//  4. infer_results    — 推理结果
// ============================================================

class DatabaseManager : public QObject
{
    Q_OBJECT

public:
    explicit DatabaseManager(QObject *parent = nullptr);
    ~DatabaseManager() override;

    // 初始化数据库（建表/升级）
    bool init(const QString &dbPath);

    bool isOpen() const;

    // ===== 训练记录 =====
    // 插入一条训练记录，返回新行的 id（失败返回 -1）
    qint64 insertTrainRecord(const QString &datasetPath,
                             const QString &modelName,
                             int epochs,
                             int batch,
                             double lr,
                             int imgsz,
                             const QString &device,
                             const QString &status,    // "completed" / "failed" / "stopped"
                             double map50,             // 训练结果 mAP50（无则传 -1）
                             double precision,         // 训练结果 Precision
                             double recall,            // 训练结果 Recall
                             const QString &outputDir,
                             const QString &bestPt = QString(),
                             const QString &notes = QString());

    // ===== 标注统计 =====
    qint64 insertAnnotationStat(const QString &imageDir,
                                int totalImages,
                                int labeledImages,
                                int totalBoxes,
                                const QString &labelList,  // JSON 字符串 ["puddle",...]
                                const QString &notes = QString());

    // ===== 量化记录 =====
    qint64 insertQuantRecord(const QString &srcModelPath,
                             const QString &quantType,    // "dynamic" / "static"
                             const QString &outputModelPath,
                             double origSizeMB,
                             double quantSizeMB,
                             const QString &status,
                             const QString &notes = QString());

    // ===== 推理结果 =====
    qint64 insertInferResult(const QString &modelPath,
                             const QString &imagePath,
                             int detectionCount,
                             double maxConfidence,
                             double inferTimeMs,
                             const QString &detectionsJson, // JSON 检测框数组
                             const QString &notes = QString());

    // ===== 查询接口 =====
    QSqlQuery queryTrainRecords(const QString &filterStatus = QString(),
                                const QDateTime &from = QDateTime(),
                                const QDateTime &to   = QDateTime());

    QSqlQuery queryAnnotationStats(const QDateTime &from = QDateTime(),
                                   const QDateTime &to   = QDateTime());

    QSqlQuery queryQuantRecords(const QDateTime &from = QDateTime(),
                                const QDateTime &to   = QDateTime());

    QSqlQuery queryInferResults(const QString &modelPath  = QString(),
                                const QDateTime &from = QDateTime(),
                                const QDateTime &to   = QDateTime());

    // ===== 统计汇总 =====
    // 返回各表的行数统计
    int countTrainRecords();
    int countAnnotationStats();
    int countQuantRecords();
    int countInferResults();

    // 推理结果：平均置信度
    double avgInferConfidence();

    // 训练记录：平均 mAP50（排除 -1 的无效值）
    double avgTrainMap50();

    QString lastError() const { return m_lastError; }

    // 调试：打印所有表的原始记录数到 qDebug
    void debugPrintAllCounts();

private:
    bool createTables();
    bool execSql(const QString &sql);

    QSqlDatabase m_db;
    QString      m_lastError;
    QString      m_connectionName;
};

#endif // DB_MANAGER_H
