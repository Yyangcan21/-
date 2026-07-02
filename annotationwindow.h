#ifndef ANNOTATIONWINDOW_H
#define ANNOTATIONWINDOW_H

#include <QMainWindow>
#include <QFileDialog>
#include <QMessageBox>
#include <QInputDialog>
#include <QProcess>
#include <QFileSystemModel>
#include <QSortFilterProxyModel>
#include <QStandardItemModel>
#include <QTableView>
#include <QModelIndex>
#include "annotationview.h"

QT_BEGIN_NAMESPACE
namespace Ui { class AnnotationWindow; }
QT_END_NAMESPACE

class AnnotationWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit AnnotationWindow(QWidget *parent = nullptr);
    ~AnnotationWindow() override;

private slots:
    void onOpenImage();
    void onOpenFolder();
    void onSave();
    void onSaveAs();
    void onImageListClicked(const QModelIndex &index);
    void onPrevImage();
    void onNextImage();
    void onAddLabel();
    void onDeleteLabel();
    void onLabelColorChange();
    void onLabelItemChanged(QStandardItem *item);
    void onSelectTool();
    void onDrawTool();
    void onAutoLabel();
    void onZoomIn()      { m_canvas->zoomIn();  }
    void onZoomOut()     { m_canvas->zoomOut(); }
    void onZoomFit()     { m_canvas->zoomFit(); }
    void onZoom100()     { m_canvas->zoom100(); }
    void onBoxesChanged(const QVector<BoundingBox> &boxes);
    void onAutoLabelReadyRead();
    void onAutoLabelFinished(int exitCode, QProcess::ExitStatus status);
    void onSelectModel();

private:
    Ui::AnnotationWindow *ui;
    AnnotationView *m_canvas;
    QLabel *m_lblInfo = nullptr;

    // 图片列表
    QStringList     m_imageFiles;
    int             m_currentIndex = -1;
    QFileSystemModel *m_fileModel = nullptr;

    // 标注数据
    QMap<QString, QVector<BoundingBox>> m_annotations;
    QString         m_currentImagePath;

    // 标签
    QStandardItemModel *m_labelModel = nullptr;
    QTableView         *m_labelTable = nullptr; // 标签表格视图
    QVector<LabelInfo>  m_labels;

    // 自动标注
    QProcess       *m_autoProcess = nullptr;
    QString         m_yoloModelPath;

    // 状态
    QString         m_currentDir;
    bool            m_dirty = false;

    // 私有工具函数
    void loadImage(const QString &path);
    void saveAnnotations();
    void loadAnnotations();
    bool askSave();
    QString imagePathToLabelPath(const QString &imgPath) const;
    void updateStatus();
    void updateImageNavigationButtons();
    void syncLabelsToCanvas();
    void addDefaultLabels();
};

#endif // ANNOTATIONWINDOW_H
