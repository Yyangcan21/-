#include "db_manager.h"
#include <QSqlRecord>
#include <QDir>
#include <QUuid>

// ============================================================
//  构造 / 析构
// ============================================================

DatabaseManager::DatabaseManager(QObject *parent)
    : QObject(parent)
{
    // 每个实例用唯一连接名，避免多实例冲突
    m_connectionName = QStringLiteral("puddle_db_") +
                       QUuid::createUuid().toString(QUuid::WithoutBraces);
}

DatabaseManager::~DatabaseManager()
{
    if (m_db.isOpen())
        m_db.close();
    QSqlDatabase::removeDatabase(m_connectionName);
}

// ============================================================
//  初始化
// ============================================================

bool DatabaseManager::init(const QString &dbPath)
{
    // 确保目录存在
    QDir dir = QFileInfo(dbPath).absoluteDir();
    if (!dir.exists())
        dir.mkpath(dir.absolutePath());

    m_db = QSqlDatabase::addDatabase(QStringLiteral("QSQLITE"), m_connectionName);
    m_db.setDatabaseName(dbPath);

    if (!m_db.open()) {
        m_lastError = m_db.lastError().text();
        qWarning() << "[DB] open failed:" << m_lastError;
        return false;
    }

    // 开启 WAL 模式，提高并发写入性能
    QSqlQuery q(m_db);
    q.exec(QStringLiteral("PRAGMA journal_mode=WAL"));
    q.exec(QStringLiteral("PRAGMA foreign_keys=ON"));

    return createTables();
}

bool DatabaseManager::isOpen() const
{
    return m_db.isValid() && m_db.isOpen();
}

// ============================================================
//  建表
// ============================================================

bool DatabaseManager::createTables()
{
    // 1. 训练记录
    if (!execSql(QStringLiteral(
        "CREATE TABLE IF NOT EXISTS train_records ("
        "  id            INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now','localtime')),"
        "  dataset_path  TEXT,"
        "  model_name    TEXT,"
        "  epochs        INTEGER,"
        "  batch         INTEGER,"
        "  lr            REAL,"
        "  imgsz         INTEGER,"
        "  device        TEXT,"
        "  status        TEXT,"     // completed / failed / stopped
        "  map50         REAL,"     // -1 if unavailable
        "  precision     REAL DEFAULT -1,"
        "  recall        REAL DEFAULT -1,"
        "  output_dir    TEXT,"
        "  best_pt       TEXT,"
        "  notes         TEXT"
        ")"
    ))) return false;

    // 兼容旧库：如果列不存在则添加
    execSql(QStringLiteral("ALTER TABLE train_records ADD COLUMN precision REAL DEFAULT -1"));
    execSql(QStringLiteral("ALTER TABLE train_records ADD COLUMN recall REAL DEFAULT -1"));
    execSql(QStringLiteral("ALTER TABLE train_records ADD COLUMN best_pt TEXT"));

    // 2. 标注统计
    if (!execSql(QStringLiteral(
        "CREATE TABLE IF NOT EXISTS annotation_stats ("
        "  id             INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now','localtime')),"
        "  image_dir      TEXT,"
        "  total_images   INTEGER,"
        "  labeled_images INTEGER,"
        "  total_boxes    INTEGER,"
        "  label_list     TEXT,"   // JSON
        "  notes          TEXT"
        ")"
    ))) return false;

    // 3. 量化记录
    if (!execSql(QStringLiteral(
        "CREATE TABLE IF NOT EXISTS quant_records ("
        "  id               INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now','localtime')),"
        "  src_model_path   TEXT,"
        "  quant_type       TEXT,"   // dynamic / static
        "  output_model_path TEXT,"
        "  orig_size_mb     REAL,"
        "  quant_size_mb    REAL,"
        "  status           TEXT,"
        "  notes            TEXT"
        ")"
    ))) return false;

    // 4. 推理结果
    if (!execSql(QStringLiteral(
        "CREATE TABLE IF NOT EXISTS infer_results ("
        "  id               INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now','localtime')),"
        "  model_path       TEXT,"
        "  image_path       TEXT,"
        "  detection_count  INTEGER,"
        "  max_confidence   REAL,"
        "  infer_time_ms    REAL,"
        "  detections_json  TEXT,"  // JSON 检测框数组
        "  notes            TEXT"
        ")"
    ))) return false;

    return true;
}

bool DatabaseManager::execSql(const QString &sql)
{
    QSqlQuery q(m_db);
    if (!q.exec(sql)) {
        m_lastError = q.lastError().text();
        qWarning() << "[DB] SQL error:" << m_lastError << "\nSQL:" << sql;
        return false;
    }
    return true;
}

// ============================================================
//  写入接口
// ============================================================

qint64 DatabaseManager::insertTrainRecord(const QString &datasetPath,
                                          const QString &modelName,
                                          int epochs, int batch, double lr,
                                          int imgsz, const QString &device,
                                          const QString &status,
                                          double map50,
                                          double precision,
                                          double recall,
                                          const QString &outputDir,
                                          const QString &bestPt,
                                          const QString &notes)
{
    QSqlQuery q(m_db);
    q.prepare(QStringLiteral(
        "INSERT INTO train_records "
        "(dataset_path,model_name,epochs,batch,lr,imgsz,device,status,map50,precision,recall,output_dir,best_pt,notes) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    ));
    q.addBindValue(datasetPath);
    q.addBindValue(modelName);
    q.addBindValue(epochs);
    q.addBindValue(batch);
    q.addBindValue(lr);
    q.addBindValue(imgsz);
    q.addBindValue(device);
    q.addBindValue(status);
    q.addBindValue(map50);
    q.addBindValue(precision);
    q.addBindValue(recall);
    q.addBindValue(outputDir);
    q.addBindValue(bestPt);
    q.addBindValue(notes);

    if (!q.exec()) {
        m_lastError = q.lastError().text();
        qWarning() << "[DB] insertTrainRecord failed:" << m_lastError;
        return -1;
    }
    return q.lastInsertId().toLongLong();
}

qint64 DatabaseManager::insertAnnotationStat(const QString &imageDir,
                                              int totalImages, int labeledImages,
                                              int totalBoxes,
                                              const QString &labelList,
                                              const QString &notes)
{
    QSqlQuery q(m_db);
    q.prepare(QStringLiteral(
        "INSERT INTO annotation_stats "
        "(image_dir,total_images,labeled_images,total_boxes,label_list,notes) "
        "VALUES (?,?,?,?,?,?)"
    ));
    q.addBindValue(imageDir);
    q.addBindValue(totalImages);
    q.addBindValue(labeledImages);
    q.addBindValue(totalBoxes);
    q.addBindValue(labelList);
    q.addBindValue(notes);

    qDebug() << "[DB] 执行 INSERT annotation_stats...";
    if (!q.exec()) {
        m_lastError = q.lastError().text();
        qWarning() << "[DB] insertAnnotationStat failed:" << m_lastError;
        return -1;
    }
    qint64 rid = q.lastInsertId().toLongLong();
    qDebug() << "[DB] INSERT 成功，rid=" << rid;
    return rid;
}

qint64 DatabaseManager::insertQuantRecord(const QString &srcModelPath,
                                          const QString &quantType,
                                          const QString &outputModelPath,
                                          double origSizeMB, double quantSizeMB,
                                          const QString &status,
                                          const QString &notes)
{
    QSqlQuery q(m_db);
    q.prepare(QStringLiteral(
        "INSERT INTO quant_records "
        "(src_model_path,quant_type,output_model_path,orig_size_mb,quant_size_mb,status,notes) "
        "VALUES (?,?,?,?,?,?,?)"
    ));
    q.addBindValue(srcModelPath);
    q.addBindValue(quantType);
    q.addBindValue(outputModelPath);
    q.addBindValue(origSizeMB);
    q.addBindValue(quantSizeMB);
    q.addBindValue(status);
    q.addBindValue(notes);

    if (!q.exec()) {
        m_lastError = q.lastError().text();
        qWarning() << "[DB] insertQuantRecord failed:" << m_lastError;
        return -1;
    }
    return q.lastInsertId().toLongLong();
}

qint64 DatabaseManager::insertInferResult(const QString &modelPath,
                                          const QString &imagePath,
                                          int detectionCount,
                                          double maxConfidence,
                                          double inferTimeMs,
                                          const QString &detectionsJson,
                                          const QString &notes)
{
    QSqlQuery q(m_db);
    q.prepare(QStringLiteral(
        "INSERT INTO infer_results "
        "(model_path,image_path,detection_count,max_confidence,infer_time_ms,detections_json,notes) "
        "VALUES (?,?,?,?,?,?,?)"
    ));
    q.addBindValue(modelPath);
    q.addBindValue(imagePath);
    q.addBindValue(detectionCount);
    q.addBindValue(maxConfidence);
    q.addBindValue(inferTimeMs);
    q.addBindValue(detectionsJson);
    q.addBindValue(notes);

    if (!q.exec()) {
        m_lastError = q.lastError().text();
        qWarning() << "[DB] insertInferResult failed:" << m_lastError;
        return -1;
    }
    return q.lastInsertId().toLongLong();
}

// ============================================================
//  查询接口
// ============================================================

QSqlQuery DatabaseManager::queryTrainRecords(const QString &filterStatus,
                                             const QDateTime &from,
                                             const QDateTime &to)
{
    QString sql = QStringLiteral("SELECT * FROM train_records WHERE 1=1");
    if (!filterStatus.isEmpty())
        sql += QStringLiteral(" AND status='%1'").arg(filterStatus);
    if (from.isValid())
        sql += QStringLiteral(" AND created_at >= '%1'").arg(from.toString(Qt::ISODate));
    if (to.isValid())
        sql += QStringLiteral(" AND created_at <= '%1'").arg(to.toString(Qt::ISODate));
    sql += QStringLiteral(" ORDER BY id DESC");

    QSqlQuery q(m_db);
    q.exec(sql);
    return q;
}

QSqlQuery DatabaseManager::queryAnnotationStats(const QDateTime &from, const QDateTime &to)
{
    QString sql = QStringLiteral("SELECT * FROM annotation_stats WHERE 1=1");
    if (from.isValid())
        sql += QStringLiteral(" AND created_at >= '%1'").arg(from.toString(Qt::ISODate));
    if (to.isValid())
        sql += QStringLiteral(" AND created_at <= '%1'").arg(to.toString(Qt::ISODate));
    sql += QStringLiteral(" ORDER BY id DESC");

    QSqlQuery q(m_db);
    q.exec(sql);
    return q;
}

QSqlQuery DatabaseManager::queryQuantRecords(const QDateTime &from, const QDateTime &to)
{
    QString sql = QStringLiteral("SELECT * FROM quant_records WHERE 1=1");
    if (from.isValid())
        sql += QStringLiteral(" AND created_at >= '%1'").arg(from.toString(Qt::ISODate));
    if (to.isValid())
        sql += QStringLiteral(" AND created_at <= '%1'").arg(to.toString(Qt::ISODate));
    sql += QStringLiteral(" ORDER BY id DESC");

    QSqlQuery q(m_db);
    q.exec(sql);
    return q;
}

QSqlQuery DatabaseManager::queryInferResults(const QString &modelPath,
                                             const QDateTime &from,
                                             const QDateTime &to)
{
    QString sql = QStringLiteral("SELECT * FROM infer_results WHERE 1=1");
    if (!modelPath.isEmpty())
        sql += QStringLiteral(" AND model_path LIKE '%%1%'").arg(modelPath);
    if (from.isValid())
        sql += QStringLiteral(" AND created_at >= '%1'").arg(from.toString(Qt::ISODate));
    if (to.isValid())
        sql += QStringLiteral(" AND created_at <= '%1'").arg(to.toString(Qt::ISODate));
    sql += QStringLiteral(" ORDER BY id DESC");

    QSqlQuery q(m_db);
    q.exec(sql);
    return q;
}

// ============================================================
//  统计汇总
// ============================================================

int DatabaseManager::countTrainRecords()
{
    QSqlQuery q(m_db);
    if (!q.exec(QStringLiteral("SELECT COUNT(*) FROM train_records"))) {
        qWarning() << "[DB] countTrainRecords failed:" << q.lastError().text();
        return 0;
    }
    return q.next() ? q.value(0).toInt() : 0;
}

int DatabaseManager::countAnnotationStats()
{
    QSqlQuery q(m_db);
    if (!q.exec(QStringLiteral("SELECT COUNT(*) FROM annotation_stats"))) {
        qWarning() << "[DB] countAnnotationStats failed:" << q.lastError().text();
        return 0;
    }
    return q.next() ? q.value(0).toInt() : 0;
}

int DatabaseManager::countQuantRecords()
{
    QSqlQuery q(m_db);
    if (!q.exec(QStringLiteral("SELECT COUNT(*) FROM quant_records"))) {
        qWarning() << "[DB] countQuantRecords failed:" << q.lastError().text();
        return 0;
    }
    return q.next() ? q.value(0).toInt() : 0;
}

int DatabaseManager::countInferResults()
{
    QSqlQuery q(m_db);
    if (!q.exec(QStringLiteral("SELECT COUNT(*) FROM infer_results"))) {
        qWarning() << "[DB] countInferResults failed:" << q.lastError().text();
        return 0;
    }
    return q.next() ? q.value(0).toInt() : 0;
}

double DatabaseManager::avgInferConfidence()
{
    QSqlQuery q(m_db);
    if (!q.exec(QStringLiteral(
        "SELECT AVG(max_confidence) FROM infer_results WHERE max_confidence > 0"
    ))) {
        qWarning() << "[DB] avgInferConfidence failed:" << q.lastError().text();
    }
    return q.next() ? q.value(0).toDouble() : 0.0;
}

void DatabaseManager::debugPrintAllCounts()
{
    QStringList tables = {
        "train_records", "annotation_stats",
        "quant_records", "infer_results"
    };
    qDebug() << "[DB Debug] ========== 原始表记录数 ==========";
    for (const QString &t : tables) {
        QSqlQuery q(m_db);
        if (!q.exec(QString("SELECT COUNT(*) FROM %1").arg(t))) {
            qWarning() << "[DB Debug]" << t << "查询失败:" << q.lastError().text();
        } else if (q.next()) {
            qDebug() << "[DB Debug]" << t << "=" << q.value(0).toInt();
        }
    }
    qDebug() << "[DB Debug] ====================================";
}

double DatabaseManager::avgTrainMap50()
{
    QSqlQuery q(m_db);
    q.exec(QStringLiteral(
        "SELECT AVG(map50) FROM train_records WHERE map50 >= 0"
    ));
    return q.next() ? q.value(0).toDouble() : 0.0;
}
