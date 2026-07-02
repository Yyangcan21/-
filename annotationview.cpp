#include "annotationview.h"
#include <QPainter>
#include <QScrollBar>
#include <QMenu>
#include <QAction>
#include <QInputDialog>
#include <QApplication>
#include <QtMath>

// ============================================================
// 颜色池：每个标签自动分配一个颜色
// ============================================================
static const QList<QColor> COLOR_POOL = {
    QColor("#FF6B6B"), QColor("#4ECDC4"), QColor("#45B7D1"),
    QColor("#96CEB4"), QColor("#FFEAA7"), QColor("#DDA0DD"),
    QColor("#98D8C8"), QColor("#F7DC6F"), QColor("#BB8FCE"),
    QColor("#85C1E9"), QColor("#F8B500"), QColor("#00CED1"),
};

static QColor nextColor(int index) {
    return COLOR_POOL[index % COLOR_POOL.size()];
}

// ============================================================
// 构造 / 析构
// ============================================================

AnnotationView::AnnotationView(QWidget *parent)
    : QWidget(parent)
{
    setFocusPolicy(Qt::StrongFocus);
    setMouseTracking(true);
    setMinimumSize(400, 300);
    // 深色背景
    setBackgroundRole(QPalette::NoRole);
}

AnnotationView::~AnnotationView() = default;

// ============================================================
// 图片管理
// ============================================================

void AnnotationView::setImage(const QImage &img)
{
    m_image = img;
    m_rotation = 0;
    rebuildPixmap();
    zoomFit();
    update();
}

void AnnotationView::clearImage()
{
    m_image = QImage();
    m_pixmap = QPixmap();
    m_boxes.clear();
    m_selectedId = -1;
    update();
}

void AnnotationView::rebuildPixmap()
{
    if (m_image.isNull()) {
        m_pixmap = QPixmap();
        return;
    }
    m_pixmap = QPixmap::fromImage(m_image);
    if (m_rotation != 0) {
        QTransform t;
        t.rotate(m_rotation);
        m_pixmap = m_pixmap.transformed(t, Qt::SmoothTransformation);
    }
}

// ============================================================
// 标注框管理
// ============================================================

void AnnotationView::setBoxes(const QVector<BoundingBox> &boxes)
{
    // 保存撤销快照
    m_undo.append(m_boxes);
    m_redo.clear();
    if (m_undo.size() > 50) m_undo.removeFirst();

    m_boxes = boxes;
    // 同步 nextId
    int maxId = 0;
    for (const auto &b : m_boxes) maxId = qMax(maxId, b.id);
    m_nextId = maxId + 1;
    update();
    emit boxesChanged(m_boxes);
}

QVector<BoundingBox> AnnotationView::getBoxes() const
{
    return m_boxes;
}

void AnnotationView::addBox(const BoundingBox &box)
{
    BoundingBox b = box;
    if (b.id < 0) b.id = m_nextId++;
    m_undo.append(m_boxes);
    m_redo.clear();
    m_boxes.append(b);
    update();
    emit boxesChanged(m_boxes);
}

void AnnotationView::removeBox(int id)
{
    m_undo.append(m_boxes);
    m_redo.clear();
    m_boxes.removeIf([id](const BoundingBox &b){ return b.id == id; });
    if (m_selectedId == id) m_selectedId = -1;
    update();
    emit boxesChanged(m_boxes);
    emitSelectedBoxInfo();
}

// ===== 新增：选中框相关（用于右侧"标注信息列表"显示 & 改类别） =====
BoundingBox AnnotationView::getSelectedBox() const
{
    BoundingBox none;
    none.id = -1;
    if (m_selectedId < 0) return none;
    for (const auto &b : m_boxes) {
        if (b.id == m_selectedId) return b;
    }
    none.label = QString();
    return none;
}

bool AnnotationView::changeSelectedBoxLabel(const QString &newLabel)
{
    if (m_selectedId < 0) return false;
    for (auto &b : m_boxes) {
        if (b.id == m_selectedId) {
            if (b.label == newLabel) return false;
            m_undo.append(m_boxes);
            m_redo.clear();
            b.label = newLabel;
            update();
            emit boxesChanged(m_boxes);
            emitSelectedBoxInfo();
            return true;
        }
    }
    return false;
}

void AnnotationView::emitSelectedBoxInfo()
{
    BoundingBox b = getSelectedBox();
    if (b.id < 0) {
        emit selectedBoxChanged(-1, QString(), 0, 0, 0, 0, 0, 0, 0, 0);
        return;
    }
    // 归一化坐标 (0~1)
    double nx = b.rect.x();
    double ny = b.rect.y();
    double nw = b.rect.width();
    double nh = b.rect.height();
    // 像素坐标（按当前 pixmap 实际尺寸算）
    int imgW = m_pixmap.width();
    int imgH = m_pixmap.height();
    int px = int(nx * imgW);
    int py = int(ny * imgH);
    int pw = int(nw * imgW);
    int ph = int(nh * imgH);
    emit selectedBoxChanged(b.id, b.label, nx, ny, nw, nh, px, py, pw, ph);
}

void AnnotationView::clearBoxes()
{
    m_undo.append(m_boxes);
    m_redo.clear();
    m_boxes.clear();
    m_selectedId = -1;
    update();
    emit boxesChanged(m_boxes);
}

// ============================================================
// 标签管理
// ============================================================

void AnnotationView::setLabels(const QVector<LabelInfo> &labels)
{
    m_labels = labels;
    update();
}

QVector<LabelInfo> AnnotationView::getLabels() const
{
    return m_labels;
}

void AnnotationView::addLabel(const QString &name, const QColor &color)
{
    LabelInfo info;
    info.name  = name;
    info.color = color.isValid() ? color : nextColor(m_labels.size());
    m_labels.append(info);
    emit currentLabelChanged(m_currentLabel);
}

void AnnotationView::removeLabel(const QString &name)
{
    m_labels.removeIf([name](const LabelInfo &l){ return l.name == name; });
    // 删除所有使用该标签的框
    bool changed = false;
    for (auto &b : m_boxes) {
        if (b.label == name) {
            b.label.clear();
            changed = true;
        }
    }
    if (changed) {
        update();
        emit boxesChanged(m_boxes);
    }
}

void AnnotationView::setCurrentLabel(const QString &label)
{
    m_currentLabel = label;
    emit currentLabelChanged(label);
    update();
}

QString AnnotationView::currentLabel() const
{
    return m_currentLabel;
}

// ============================================================
// 缩放 / 平移
// ============================================================

void AnnotationView::zoomIn()    { zoomBy(1.25); }
void AnnotationView::zoomOut()   { zoomBy(0.8);  }

void AnnotationView::zoomBy(double factor)
{
    if (m_image.isNull()) return;
    QPointF center(width()/2.0, height()/2.0);
    double newScale = qBound(0.02, m_scale * factor, 50.0);
    m_offset = center - (center - m_offset) * (newScale / m_scale);
    m_scale = newScale;
    update();
}

void AnnotationView::zoomFit()
{
    if (m_image.isNull() || width() <= 0 || height() <= 0) return;
    double pixW = m_pixmap.width();
    double pixH = m_pixmap.height();
    if (pixW <= 0 || pixH <= 0) return;
    double sx = double(width())  / pixW;
    double sy = double(height()) / pixH;
    m_scale = qMin(sx, sy) * 0.95;
    m_offset = QPointF(
        (width()  - pixW * m_scale) / 2.0,
        (height() - pixH * m_scale) / 2.0
    );
    update();
}

void AnnotationView::zoom100()
{
    if (m_image.isNull()) return;
    double pixW = m_pixmap.width();
    double pixH = m_pixmap.height();
    QPointF center(width()/2.0, height()/2.0);
    m_offset = center - QPointF(pixW, pixH) * 0.5;
    m_scale = 1.0;
    update();
}

void AnnotationView::panBy(const QPointF &delta)
{
    m_offset += delta;
    update();
}

// ---- 图像旋转 ----

static QRectF rotateBox90(const QRectF &r) {
    QRectF n = r.normalized();
    return QRectF(1.0 - n.bottom(), n.left(), n.height(), n.width());
}

static QRectF rotateBox270(const QRectF &r) {
    QRectF n = r.normalized();
    return QRectF(n.top(), 1.0 - n.right(), n.height(), n.width());
}

void AnnotationView::rotateCW()
{
    if (m_image.isNull()) return;
    // 保存撤销
    m_undo.append(m_boxes);
    m_redo.clear();
    if (m_undo.size() > 50) m_undo.removeFirst();

    // 变换所有框坐标
    for (auto &b : m_boxes)
        b.rect = rotateBox90(b.rect);

    m_rotation = (m_rotation + 90) % 360;
    m_selectedId = -1;
    rebuildPixmap();
    zoomFit();
    emit boxesChanged(m_boxes);
    emit rotationChanged(m_rotation);
}

void AnnotationView::rotateCCW()
{
    if (m_image.isNull()) return;
    m_undo.append(m_boxes);
    m_redo.clear();
    if (m_undo.size() > 50) m_undo.removeFirst();

    for (auto &b : m_boxes)
        b.rect = rotateBox270(b.rect);

    m_rotation = (m_rotation + 270) % 360; // -90 等价
    m_selectedId = -1;
    rebuildPixmap();
    zoomFit();
    emit boxesChanged(m_boxes);
    emit rotationChanged(m_rotation);
}

void AnnotationView::rotateReset()
{
    if (m_rotation == 0) return;
    while (m_rotation != 0)
        rotateCW();
}

void AnnotationView::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    zoomFit();
}

// ============================================================
// 坐标转换
// ============================================================

QPointF AnnotationView::toImage(const QPointF &wp) const
{
    if (m_image.isNull()) return QPointF();
    // 使用 pixmap 尺寸（旋转后可能不同）
    double pw = m_pixmap.width();
    double ph = m_pixmap.height();
    if (pw <= 0 || ph <= 0) return QPointF();
    return QPointF(
        (wp.x() - m_offset.x()) / m_scale / pw,
        (wp.y() - m_offset.y()) / m_scale / ph
    );
}

QPointF AnnotationView::toWidget(const QPointF &ip) const
{
    double pw = m_pixmap.width();
    double ph = m_pixmap.height();
    return QPointF(
        ip.x() * pw * m_scale + m_offset.x(),
        ip.y() * ph * m_scale + m_offset.y()
    );
}

QRectF AnnotationView::toWidgetRect(const QRectF &r) const
{
    return QRectF(toWidget(r.topLeft()), toWidget(r.bottomRight())).normalized();
}

// ============================================================
// 命中测试
// ============================================================

int AnnotationView::hitTestBox(const QPointF &pt) const
{
    // 从上到下遍历（后面画的在上层）
    // 扩大 3px 容差，确保边角点击能被捕获
    constexpr double kMargin = 3.0;
    for (int i = m_boxes.size() - 1; i >= 0; --i) {
        QRectF wr = toWidgetRect(m_boxes[i].rect).adjusted(-kMargin, -kMargin, kMargin, kMargin);
        if (wr.contains(pt)) return m_boxes[i].id;
    }
    return -1;
}

int AnnotationView::hitTestHandle(const BoundingBox &box, const QPointF &pt) const
{
    QRectF wr = toWidgetRect(box.rect);
    // 8个控制点：0左上 1上中 2右上 3右中 4右下 5下中 6左下 7左中（顺时针）
    QList<QPointF> handles = {
        wr.topLeft(),     wr.topLeft() + QPointF(wr.width()/2, 0),
        wr.topRight(),    wr.topRight() + QPointF(0, wr.height()/2),
        wr.bottomRight(), wr.bottomRight() - QPointF(wr.width()/2, 0),
        wr.bottomLeft(),  wr.bottomLeft() - QPointF(0, wr.height()/2)
    };
    // 使用稍大命中半径，便于在小缩放比例下操作
    constexpr double kHandleHitR = 10.0;
    for (int i = 0; i < handles.size(); ++i) {
        if (QLineF(pt, handles[i]).length() <= kHandleHitR)
            return i;
    }
    return -1;
}

// ============================================================
// 绘图
// ============================================================

void AnnotationView::paintEvent(QPaintEvent *)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    // 背景
    p.fillRect(rect(), QColor("#1e1e1e"));

    if (m_image.isNull()) {
        p.setPen(Qt::white);
        p.drawText(rect(), Qt::AlignCenter, "请加载图片\n（左上角「打开图片」或「打开文件夹」）");
        return;
    }

    // 绘制图片（使用缩放后的 pixmap 加速）
    double dispW = m_pixmap.width()  * m_scale;
    double dispH = m_pixmap.height() * m_scale;
    p.drawPixmap(m_offset, m_pixmap.scaled(
        dispW, dispH, Qt::KeepAspectRatio, Qt::SmoothTransformation));

    // 绘制网格（可选）
    if (m_scale > 0.5) {
        p.setPen(QPen(QColor(255,255,255,30), 1));
        int pw = m_pixmap.width();
        int ph = m_pixmap.height();
        for (int x = 0; x < pw; x += 100) {
            double wx = x * m_scale + m_offset.x();
            p.drawLine(QPointF(wx, m_offset.y()),
                       QPointF(wx, dispH + m_offset.y()));
        }
        for (int y = 0; y < ph; y += 100) {
            double wy = y * m_scale + m_offset.y();
            p.drawLine(QPointF(m_offset.x(), wy),
                       QPointF(dispW + m_offset.x(), wy));
        }
    }

    // 绘制所有标注框
    for (const auto &box : m_boxes) {
        // 查找标签颜色
        QColor boxColor = Qt::red;
        for (const auto &lbl : m_labels) {
            if (lbl.name == box.label) {
                boxColor = lbl.color;
                if (!lbl.visible) { boxColor.setAlpha(60); }
                break;
            }
        }
        drawBox(p, box, boxColor);
    }

    // 绘制正在画的新框
    if (m_interactMode == Drawing) {
        QPen pen(QColor(255, 200, 0), 2, Qt::DashLine);
        p.setPen(pen);
        p.setBrush(QColor(255, 200, 0, 40));
        p.drawRect(toWidgetRect(m_drawRect));

        // 显示尺寸信息
        if (m_drawRect.width() > 0.02 && m_drawRect.height() > 0.02) {
            QRectF wr = toWidgetRect(m_drawRect);
            p.setPen(Qt::yellow);
            p.setFont(QFont("Consolas", 10));
            p.drawText(wr.bottomRight() + QPointF(4, 0),
                       QString("%1×%2")
                           .arg(int(m_drawRect.width() * m_pixmap.width()))
                           .arg(int(m_drawRect.height() * m_pixmap.height())));
        }
    }

    // 当前鼠标十字线
    {
        QPointF normPt = toImage(mapFromGlobal(QCursor::pos()));
        if (normPt.x() >= 0 && normPt.x() <= 1 && normPt.y() >= 0 && normPt.y() <= 1) {
            QPointF mp = mapFromGlobal(QCursor::pos());
            if (m_scale > 0.3) drawCrosshair(p, mp);
        }
    }

    // 调试：左上角显示当前工具模式 + 框数量
    p.setPen(QColor(255, 255, 0));
    p.setFont(QFont("Consolas", 11, QFont::Bold));
    p.drawText(8, 18, QString("MODE=%1  BOXES=%2  SEL=%3")
        .arg(m_tool == TOOL_DRAW ? "DRAW" : "SELECT")
        .arg(m_boxes.size())
        .arg(m_selectedId));
}

void AnnotationView::drawBox(QPainter &p, const BoundingBox &box, const QColor &color)
{
    BoundingBox b = box;
    // 临时注入颜色查找逻辑
    QRectF wr = toWidgetRect(b.rect);
    bool selected = (b.id == m_selectedId);
    // 不再用 fillRect 半透明覆盖图片——避免视觉上"框消失"的错觉
    QPen pen(color, selected ? 3 : 2);
    p.setPen(pen);
    p.setBrush(Qt::NoBrush);
    p.drawRect(wr);

    if (selected) {
        p.setPen(Qt::white);
        p.setBrush(color);
        QList<QPointF> handles = {
            wr.topLeft(), wr.topLeft() + QPointF(wr.width()/2, 0),
            wr.topRight(), wr.topRight() + QPointF(0, wr.height()/2),
            wr.bottomRight(), wr.bottomRight() - QPointF(wr.width()/2, 0),
            wr.bottomLeft(), wr.bottomLeft() - QPointF(0, wr.height()/2)
        };
        for (const auto &h : handles)
            p.drawEllipse(h, HANDLE_R, HANDLE_R);
    }

    if (!b.label.isEmpty()) {
        p.setFont(QFont("Microsoft YaHei", 12, QFont::Bold));
        QRectF textBg = wr.adjusted(0, -22, 0, 0);
        QColor bg = color.darker(120);
        bg.setAlpha(200);
        p.fillRect(textBg, bg);
        p.setPen(Qt::white);
        p.drawText(textBg, Qt::AlignCenter, b.label);
    }
}

void AnnotationView::drawCrosshair(QPainter &p, const QPointF &pt)
{
    p.setPen(QPen(QColor(255,255,255,80), 1, Qt::DashLine));
    p.drawLine(pt.x(), 0, pt.x(), height());
    p.drawLine(0, pt.y(), width(), pt.y());
}

// ============================================================
// 鼠标事件
// ============================================================

void AnnotationView::mousePressEvent(QMouseEvent *event)
{
    if (m_image.isNull()) return;

    QPointF wp   = event->pos();
    QPointF norm = toImage(wp);

    // 中键 或 Shift+左键 → 平移
    if (event->button() == Qt::MiddleButton ||
        (event->button() == Qt::LeftButton && event->modifiers() & Qt::ShiftModifier)) {
        m_panStart = wp;
        setCursor(Qt::SizeAllCursor);
        return;
    }

    if (event->button() != Qt::LeftButton) return;

    // ====================
    // 画框模式
    // ====================
    if (m_tool == TOOL_DRAW) {
        // 起点必须在图片范围内，否则点空白处会出现"红点跑到左上角"的诡异效果
        if (norm.x() < 0 || norm.x() > 1 || norm.y() < 0 || norm.y() > 1) {
            qDebug() << "[PRESS] DRAW ignored (out of image)";
            return;
        }
        m_interactMode = Drawing;
        m_drawRect     = QRectF(norm, norm);
        qDebug() << "[PRESS] DRAW start, pos=" << norm;
        update();
        return;
    }

    // ====================
    // 选择模式：手柄检测优先于框内部检测
    // ====================
    m_interactMode = Idle;
    m_resizeHandle = -1;

    // ---- 第一遍：从上到下检测所有框的手柄（高优先级） ----
    for (int i = m_boxes.size() - 1; i >= 0; --i) {
        int h = hitTestHandle(m_boxes[i], wp);
        if (h >= 0) {
            m_interactMode   = Resizing;
            m_selectedId     = m_boxes[i].id;
            m_resizeHandle   = h;
            m_resizeOrigRect = m_boxes[i].rect;
            m_dragStartNorm  = norm;
            for (auto &b2 : m_boxes) b2.selected = (b2.id == m_boxes[i].id);
            qDebug() << "[PRESS] RESIZE start, id=" << m_selectedId << "handle=" << h;
            update();
            return;
        }
    }

    // ---- 第二遍：检测框内部（扩大容差后的命中区域） ----
    int id = hitTestBox(wp);
    if (id >= 0) {
        m_interactMode  = Dragging;
        m_selectedId    = id;
        m_dragStartNorm = norm;
        for (auto &b2 : m_boxes) b2.selected = (b2.id == id);
        emitSelectedBoxInfo();
        qDebug() << "[PRESS] DRAG start, id=" << id << "pos=" << norm;
        update();
        return;
    }

    // ---- 点击空白 → 取消选中 ----
    qDebug() << "[PRESS] miss → deselect all";
    m_selectedId = -1;
    for (auto &b : m_boxes) b.selected = false;
    update();
    emitSelectedBoxInfo();
}

void AnnotationView::mouseMoveEvent(QMouseEvent *event)
{
    if (m_image.isNull()) return;

    QPointF wp   = event->pos();
    QPointF norm = toImage(wp);

    // 平移
    if (event->buttons() & Qt::MiddleButton ||
        (event->buttons() & Qt::LeftButton && event->modifiers() & Qt::ShiftModifier)) {
        m_offset   += (wp - m_panStart);
        m_panStart  = wp;
        update();
        return;
    }

    // ---------- 画框 ----------
    if (m_interactMode == Drawing) {
        QPointF end = norm;
        // 钳制终点在图片内
        end.setX(qBound(0.0, end.x(), 1.0));
        end.setY(qBound(0.0, end.y(), 1.0));
        m_drawRect = QRectF(m_drawRect.topLeft(), end).normalized();
        update();
        return;
    }

    // ---------- 拖拽 ----------
    if (m_interactMode == Dragging && m_selectedId >= 0) {
        for (auto &b : m_boxes) {
            if (b.id != m_selectedId) continue;

            QPointF delta = norm - m_dragStartNorm;
            QRectF nr = b.rect.normalized();
            nr.translate(delta);

            // 边界钳制：不允许框超出 [0, 1] 范围
            double w = nr.width();
            double h = nr.height();
            double x = qBound(0.0, nr.left(), 1.0 - w);
            double y = qBound(0.0, nr.top(),  1.0 - h);
            b.rect = QRectF(x, y, w, h);

            m_dragStartNorm = norm;  // 增量式
            break;
        }
        update();
        return;
    }

    // ---------- 缩放 ----------
    if (m_interactMode == Resizing && m_selectedId >= 0) {
        for (auto &b : m_boxes) {
            if (b.id != m_selectedId) continue;

            QRectF r = m_resizeOrigRect.normalized();
            QPointF d = norm - m_dragStartNorm;

            // 8个方向：0左上 1上 2右上 3右 4右下 5下 6左下 7左
            switch (m_resizeHandle) {
            case 0: r = QRectF(r.left()+d.x(), r.top()+d.y(), r.width()-d.x(), r.height()-d.y()); break;
            case 1: r = QRectF(r.left(),       r.top()+d.y(), r.width(),     r.height()-d.y()); break;
            case 2: r = QRectF(r.left(),       r.top()+d.y(), r.width()+d.x(), r.height()-d.y()); break;
            case 3: r = QRectF(r.left(),       r.top(),       r.width()+d.x(), r.height());     break;
            case 4: r = QRectF(r.left(),       r.top(),       r.width()+d.x(), r.height()+d.y()); break;
            case 5: r = QRectF(r.left(),       r.top(),       r.width(),     r.height()+d.y()); break;
            case 6: r = QRectF(r.left()+d.x(), r.top(),       r.width()-d.x(), r.height()+d.y()); break;
            case 7: r = QRectF(r.left()+d.x(), r.top(),       r.width()-d.x(), r.height());     break;
            }

            r = r.normalized();

            // 钳制在 [0,1] 内，最小尺寸 3px（归一化约 0.003）
            constexpr double kMinNorm = 0.003;
            double cx = qBound(0.0, r.left(), 1.0 - kMinNorm);
            double cy = qBound(0.0, r.top(),  1.0 - kMinNorm);
            double cw = qMax(kMinNorm, r.width());
            double ch = qMax(kMinNorm, r.height());
            // 防止超出右/下边界
            if (cx + cw > 1.0) cw = 1.0 - cx;
            if (cy + ch > 1.0) ch = 1.0 - cy;
            b.rect = QRectF(cx, cy, cw, ch);
            break;
        }
        update();
        return;
    }

    // ---------- 空闲时更新鼠标形状 ----------
    // 手柄优先检测
    for (int i = m_boxes.size() - 1; i >= 0; --i) {
        int h = hitTestHandle(m_boxes[i], wp);
        if (h >= 0) {
            static const Qt::CursorShape cs[8] = {
                Qt::SizeFDiagCursor, Qt::SizeVerCursor, Qt::SizeBDiagCursor,
                Qt::SizeHorCursor,  Qt::SizeFDiagCursor, Qt::SizeVerCursor,
                Qt::SizeBDiagCursor, Qt::SizeHorCursor
            };
            setCursor(cs[h]);
            return;
        }
    }

    // 框内部
    int hid = hitTestBox(wp);
    if (hid >= 0) {
        setCursor(m_tool == TOOL_DRAW ? Qt::CrossCursor : Qt::SizeAllCursor);
        return;
    }

    setCursor(m_tool == TOOL_DRAW ? Qt::CrossCursor : Qt::ArrowCursor);
}

void AnnotationView::mouseReleaseEvent(QMouseEvent *event)
{
    if (event->button() != Qt::LeftButton) return;

    // ---------- 完成画框 ----------
    if (m_interactMode == Drawing) {
        m_interactMode = Idle;
        if (m_drawRect.width() > 0.01 && m_drawRect.height() > 0.01) {
            BoundingBox newBox;
            newBox.id    = m_nextId++;
            newBox.rect  = m_drawRect.normalized();
            newBox.label = m_currentLabel;
            addBox(newBox);
            // ★ 画完框立刻选中它，这样 8 个控制点和粗边线才会显示出来
            m_selectedId = newBox.id;
            for (auto &b2 : m_boxes) b2.selected = (b2.id == newBox.id);
            emitSelectedBoxInfo();
            qDebug() << "[RELEASE] DRAW finished, id=" << newBox.id
                     << "rect=" << newBox.rect;
        } else {
            qDebug() << "[RELEASE] DRAW skipped (too small) w=" << m_drawRect.width()
                     << "h=" << m_drawRect.height();
        }
        m_drawRect = QRectF();
        // 画完框自动切回选择模式，方便立即移动/调整刚画的框
        if (m_tool != TOOL_SELECT) {
            m_tool = TOOL_SELECT;
            emit toolChanged(TOOL_SELECT);
            qDebug() << "[RELEASE] auto-switch to SELECT";
        }
        update();
        return;
    }

    // ---------- 完成拖拽 ----------
    if (m_interactMode == Dragging) {
        m_interactMode = Idle;
        emit boxesChanged(m_boxes);
        qDebug() << "[RELEASE] DRAG finished";
        return;
    }

    // ---------- 完成缩放 ----------
    if (m_interactMode == Resizing) {
        m_interactMode  = Idle;
        m_resizeHandle  = -1;
        emit boxesChanged(m_boxes);
        qDebug() << "[RELEASE] RESIZE finished";
        return;
    }

    // 平移结束
    if (event->button() == Qt::MiddleButton ||
        (event->button() == Qt::LeftButton && event->modifiers() & Qt::ShiftModifier)) {
        setCursor(Qt::ArrowCursor);
    }
}

void AnnotationView::mouseDoubleClickEvent(QMouseEvent *event)
{
    // 双击框：进入编辑标签
    if (event->button() == Qt::LeftButton) {
        int id = hitTestBox(event->pos());
        if (id >= 0) {
            QString newLabel = QInputDialog::getText(this, "修改标签", "请输入标签名称：");
            if (!newLabel.isEmpty()) {
                for (auto &b : m_boxes) {
                    if (b.id == id) {
                        b.label = newLabel;
                        break;
                    }
                }
                update();
                emit boxesChanged(m_boxes);
            }
        }
    }
}

void AnnotationView::wheelEvent(QWheelEvent *event)
{
    if (event->modifiers() & Qt::ControlModifier) {
        double factor = event->angleDelta().y() > 0 ? 1.15 : 1.0/1.15;
        QPointF center = event->position();
        double newScale = qBound(0.02, m_scale * factor, 50.0);
        m_offset = center - (center - m_offset) * (newScale / m_scale);
        m_scale = newScale;
        update();
        event->accept();
    } else {
        // 滚轮平移
        m_offset += QPointF(-event->angleDelta().x(), -event->angleDelta().y()) * 0.5;
        update();
        event->accept();
    }
}

void AnnotationView::keyPressEvent(QKeyEvent *event)
{
    switch (event->key()) {
        case Qt::Key_Delete:
        case Qt::Key_Backspace:
            if (m_selectedId >= 0) {
                removeBox(m_selectedId);
                m_selectedId = -1;
            }
            break;
        case Qt::Key_Z:
            if (event->modifiers() & Qt::ControlModifier) undo();
            break;
        case Qt::Key_Y:
            if (event->modifiers() & Qt::ControlModifier) redo();
            break;
        case Qt::Key_A:
            if (event->modifiers() & Qt::ControlModifier) {
                for (auto &b : m_boxes) b.selected = true;
                update();
            }
            break;
        case Qt::Key_Escape:
            m_selectedId    = -1;
            m_interactMode  = Idle;
            m_drawRect      = QRectF();
            for (auto &b : m_boxes) b.selected = false;
            update();
            emitSelectedBoxInfo();
            break;
        case Qt::Key_S:
            if (event->modifiers() & Qt::ControlModifier)
                emit boxesChanged(m_boxes); // 触发保存
            break;
        case Qt::Key_R:
            if (event->modifiers() & Qt::ControlModifier)
                rotateCW();
            break;
    }
}

void AnnotationView::contextMenuEvent(QContextMenuEvent *event)
{
    QMenu menu(this);
    QAction *delAct = menu.addAction("🗑 删除选中框");
    QAction *clsAct = menu.addAction("🔄 清空全部标注");
    menu.addSeparator();
    QAction *zoomFitAct  = menu.addAction("🔲 适应窗口");
    QAction *zoom100Act  = menu.addAction("🔍 100%");
    menu.addSeparator();
    QAction *rotCWAct    = menu.addAction("↻ 顺时针旋转");
    QAction *rotCCWAct   = menu.addAction("↺ 逆时针旋转");
    QAction *rotRstAct   = menu.addAction("↕ 重置旋转");
    menu.addSeparator();
    QAction *lblAct = menu.addAction("🏷 为选中框设置标签...");

    QAction *sel = menu.exec(event->globalPos());
    if (!sel) return;

    if (sel == delAct) {
        if (m_selectedId >= 0) removeBox(m_selectedId), m_selectedId = -1;
    } else if (sel == clsAct) {
        clearBoxes();
    } else if (sel == zoomFitAct) {
        zoomFit();
    } else if (sel == zoom100Act) {
        zoom100();
    } else if (sel == rotCWAct) {
        rotateCW();
    } else if (sel == rotCCWAct) {
        rotateCCW();
    } else if (sel == rotRstAct) {
        rotateReset();
    } else if (sel == lblAct) {
        if (m_selectedId >= 0) {
            QStringList names;
            for (const auto &l : m_labels) names << l.name;
            bool ok;
            QString text = QInputDialog::getItem(this, "设置标签",
                "选择或输入标签：", names, 0, true, &ok);
            if (ok && !text.isEmpty()) {
                for (auto &b : m_boxes)
                    if (b.id == m_selectedId) { b.label = text; break; }
                update();
                emit boxesChanged(m_boxes);
            }
        }
    }
}

// ============================================================
// 撤销 / 重做
// ============================================================

void AnnotationView::undo()
{
    if (m_undo.isEmpty()) return;
    m_redo.append(m_boxes);
    m_boxes = m_undo.takeLast();
    m_selectedId = -1;
    update();
    emit boxesChanged(m_boxes);
}

void AnnotationView::redo()
{
    if (m_redo.isEmpty()) return;
    m_undo.append(m_boxes);
    m_boxes = m_redo.takeLast();
    m_selectedId = -1;
    update();
    emit boxesChanged(m_boxes);
}

bool AnnotationView::canUndo() const { return !m_undo.isEmpty(); }
bool AnnotationView::canRedo() const { return !m_redo.isEmpty(); }
