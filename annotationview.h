#ifndef ANNOTATIONVIEW_H
#define ANNOTATIONVIEW_H

#include <QWidget>
#include <QMouseEvent>
#include <QPainter>
#include <QWheelEvent>
#include <QKeyEvent>
#include <QRectF>
#include <QPointF>
#include <QVector>
#include <QSet>
#include <QMap>
#include <QDebug>

// 单个标注框（归一化坐标 [0,1]）
struct BoundingBox {
    int         id;
    QString     label;
    QRectF      rect;   // normalized [0,1]
    bool        selected = false;
    QPointF     dragStart;    // 拖拽起点（归一化）
    bool        resizing = false;
    int         resizeHandle = -1; // 0-7 八个控制点
};

// 标签信息
struct LabelInfo {
    QString name;
    QColor  color;
    bool    visible = true;
};

class AnnotationView : public QWidget {
    Q_OBJECT
public:
    enum Tool { TOOL_SELECT, TOOL_DRAW };

    explicit AnnotationView(QWidget *parent = nullptr);
    ~AnnotationView() override;

    void setImage(const QImage &img);
    void clearImage();
    QImage getImage() const { return m_image; }

    void setBoxes(const QVector<BoundingBox> &boxes);
    QVector<BoundingBox> getBoxes() const;
    void addBox(const BoundingBox &box);
    void removeBox(int id);
    void clearBoxes();

    void setLabels(const QVector<LabelInfo> &labels);
    QVector<LabelInfo> getLabels() const;
    void addLabel(const QString &name, const QColor &color);
    void removeLabel(const QString &name);
    void setCurrentLabel(const QString &label);
    QString currentLabel() const;

    // ===== 选中框相关（用于右侧"标注信息列表"显示 & 改类别） =====
    int  getSelectedId() const { return m_selectedId; }
    BoundingBox getSelectedBox() const;            // 返回当前选中的框（没有就返回 id=-1）
    bool changeSelectedBoxLabel(const QString &newLabel); // 改当前选中框的 label
    void emitSelectedBoxInfo();                    // 主动发一次 selectedBoxChanged（外部刷新列表时用）
    void setTool(Tool tool) {
        qDebug() << "[setTool] from" << m_tool << "to" << tool;
        // 切换工具时，如果有未完成的框先保存下来
        if (tool != m_tool && m_interactMode == Drawing && m_drawRect.width() > 0.01 && m_drawRect.height() > 0.01) {
            BoundingBox newBox;
            newBox.id    = m_nextId++;
            newBox.rect  = m_drawRect.normalized();
            newBox.label = m_currentLabel;
            m_boxes.append(newBox);
        }
        m_interactMode = Idle;
        m_drawRect = QRectF();
        m_tool = tool;
        update();
    }

    void zoomIn();
    void zoomOut();
    void zoomFit();
    void zoom100();
    void zoomBy(double factor);
    void panBy(const QPointF &delta);

    // 图像旋转
    void rotateCW();        // 顺时针90°
    void rotateCCW();       // 逆时针90°
    void rotateReset();     // 重置为0°
    int  rotation() const { return m_rotation; }

    void undo();
    void redo();
    bool canUndo() const;
    bool canRedo() const;

signals:
    void boxesChanged(const QVector<BoundingBox> &boxes);
    void currentLabelChanged(const QString &label);
    void rotationChanged(int angle);
    void toolChanged(Tool tool);   // 工具切换时通知外部同步 UI 状态
    void selectedBoxChanged(int boxId, const QString &label,
                            double x, double y, double w, double h,
                            int pixelX, int pixelY, int pixelW, int pixelH);
                            // 选中框变化时通知外部（用于更新右侧信息列表）

protected:
    void paintEvent(QPaintEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void mouseDoubleClickEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;
    void contextMenuEvent(QContextMenuEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;

private:
    // 坐标转换
    QPointF toImage(const QPointF &wp) const;    // widget -> normalized image
    QPointF toWidget(const QPointF &ip) const;    // normalized image -> widget
    QRectF  toWidgetRect(const QRectF &r) const;

    // 命中测试
    int  hitTestBox(const QPointF &pt) const;
    int  hitTestHandle(const BoundingBox &box, const QPointF &pt) const;

    // 绘制
    void drawBox(QPainter &p, const BoundingBox &box, const QColor &color);
    void drawCrosshair(QPainter &p, const QPointF &pt);
    void rebuildPixmap();

    Tool m_tool = TOOL_SELECT;

    QImage  m_image;
    QPixmap m_pixmap;

    // 视图
    QPointF m_offset;   // 图片左上角在 widget 中的位置（像素）
    double  m_scale = 1.0;
    QPointF m_panStart;

    // 标注
    QVector<BoundingBox> m_boxes;
    int m_nextId = 0;
    QString m_currentLabel;
    QVector<LabelInfo> m_labels;

    // ===== 鼠标交互状态机 =====
    enum InteractMode { Idle, Drawing, Dragging, Resizing };
    InteractMode m_interactMode = Idle;

    // 画框
    QRectF m_drawRect; // normalized

    // 选中哪个框
    int    m_selectedId = -1;

    // 拖拽/缩放起点（normalized坐标）
    QPointF m_dragStartNorm;

    // 缩放专用
    int    m_resizeHandle = -1;     // 0~7
    QRectF m_resizeOrigRect;        // 缩放前的原始rect

    // 撤销栈
    QVector<QVector<BoundingBox>> m_undo, m_redo;

    // 旋转
    int m_rotation = 0;   // 0, 90, 180, 270

    static constexpr int HANDLE_R = 6; // 控制点像素半径
};

#endif // ANNOTATIONVIEW_H

