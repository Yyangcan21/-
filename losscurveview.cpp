#include "losscurveview.h"
#include <QPainter>
#include <QPainterPath>
#include <QFontMetrics>
#include <algorithm>
#include <cmath>

LossCurveView::LossCurveView(QWidget *parent)
    : QWidget(parent)
{
    // 2 条曲线：cls + dfl
    m_curves.append({"cls_loss",     QColor("#4ECDC4"), {}});
    m_curves.append({"dfl_loss",     QColor("#FFD93D"), {}});

    setMinimumHeight(180);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
}

void LossCurveView::clearCurves()
{
    for (auto &c : m_curves) c.points.clear();
    update();
}

void LossCurveView::addDataPoint(int epoch,
                                 double clsLoss, double dflLoss,
                                 double map50, double precision, double recall)
{
    double vals[2] = { clsLoss, dflLoss };
    for (int i = 0; i < 2; ++i) {
        if (vals[i] >= 0.0) {  // 未知值传 -1
            m_curves[i].points.append(QPointF(epoch, vals[i]));
        }
    }
    if (epoch > m_maxEpoch) m_maxEpoch = epoch;
    update();
}

void LossCurveView::paintEvent(QPaintEvent *)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    // ===== 1. 背景 =====
    p.fillRect(rect(), QColor("#1e1e1e"));

    // 边距：左边留给 Y 轴刻度，下边留 X 轴刻度，右上角留给图例
    const int marginL = 50;
    const int marginR = 20;
    const int marginT = 30;   // 顶部留 30 像素给图例
    const int marginB = 30;

    QRectF plot(marginL, marginT,
                width() - marginL - marginR,
                height() - marginT - marginB);

    p.fillRect(plot, QColor("#252525"));

    // ===== 2. 计算 Y 轴范围（自动） =====
    double yMin = 0.0, yMax = 1.0;
    bool anyData = false;
    for (const auto &c : m_curves) {
        for (const auto &pt : c.points) {
            anyData = true;
            yMin = std::min(yMin, pt.y());
            yMax = std::max(yMax, pt.y());
        }
    }
    if (anyData) {
        // 加一点 padding
        double pad = (yMax - yMin) * 0.1;
        if (pad < 0.01) pad = 0.01;
        yMin = std::max(0.0, yMin - pad);
        yMax = yMax + pad;
    } else {
        yMin = 0.0;
        yMax = 1.0;
    }

    // ===== 3. 网格 =====
    p.setPen(QPen(QColor(80, 80, 80), 1, Qt::SolidLine));
    p.setFont(QFont("Consolas", 8));
    int yTicks = 5;
    for (int i = 0; i <= yTicks; ++i) {
        double ratio = double(i) / yTicks;
        double yv = yMin + (yMax - yMin) * (1.0 - ratio);
        double py = plot.top() + plot.height() * ratio;
        p.drawLine(QPointF(plot.left(), py), QPointF(plot.right(), py));
        // Y 轴刻度文字
        p.setPen(QColor(180, 180, 180));
        p.drawText(QRectF(0, py - 8, marginL - 4, 16),
                   Qt::AlignRight | Qt::AlignVCenter,
                   QString::number(yv, 'f', 2));
        p.setPen(QPen(QColor(80, 80, 80), 1, Qt::SolidLine));
    }

    // X 轴刻度
    int xTicks = std::min(m_maxEpoch, 10);
    if (xTicks < 2) xTicks = 2;
    for (int i = 0; i <= xTicks; ++i) {
        int epoch = int(double(m_maxEpoch) * i / xTicks);
        double px = plot.left() + plot.width() * double(i) / xTicks;
        p.drawLine(QPointF(px, plot.top()), QPointF(px, plot.bottom()));
        p.setPen(QColor(180, 180, 180));
        p.drawText(QRectF(px - 20, plot.bottom() + 4, 40, 16),
                   Qt::AlignCenter, QString::number(epoch));
        p.setPen(QPen(QColor(80, 80, 80), 1, Qt::SolidLine));
    }

    // ===== 4. 画 2 条曲线 =====
    auto toPx = [&](const QPointF &pt) -> QPointF {
        double x = plot.left() + plot.width() * (pt.x() - 1) / m_maxEpoch;
        double y = plot.top() + plot.height() * (1.0 - (pt.y() - yMin) / (yMax - yMin));
        return QPointF(x, y);
    };

    for (const auto &c : m_curves) {
        if (c.points.size() < 1) continue;
        // 折线
        QPainterPath path;
        QPointF start = toPx(c.points.first());
        path.moveTo(start);
        for (int i = 1; i < c.points.size(); ++i) {
            path.lineTo(toPx(c.points[i]));
        }
        p.setPen(QPen(c.color, 2));
        p.setBrush(Qt::NoBrush);
        p.drawPath(path);

        // 数据点
        p.setBrush(c.color);
        p.setPen(Qt::NoPen);
        for (const auto &pt : c.points) {
            p.drawEllipse(toPx(pt), 3, 3);
        }
    }

    // ===== 5. 边框 =====
    p.setPen(QPen(QColor(120, 120, 120), 1));
    p.setBrush(Qt::NoBrush);
    p.drawRect(plot);

    // ===== 6. 图例（右上角） =====
    int lx = plot.right() - 130;
    int ly = plot.top() + 4;
    p.setBrush(QColor(0, 0, 0, 160));
    p.setPen(QPen(QColor(100, 100, 100), 1));
    p.drawRoundedRect(QRectF(lx - 4, ly - 2, 134, 16 * m_curves.size() + 4), 3, 3);

    p.setFont(QFont("Consolas", 8));
    for (int i = 0; i < m_curves.size(); ++i) {
        int yy = ly + i * 16 + 8;
        p.setPen(m_curves[i].color);
        p.setBrush(m_curves[i].color);
        p.drawRect(lx, yy - 4, 10, 3);
        p.setPen(QColor(220, 220, 220));
        p.drawText(QRectF(lx + 14, yy - 8, 110, 14),
                   Qt::AlignVCenter | Qt::AlignLeft,
                   m_curves[i].name);
    }

    // ===== 7. 标题 =====
    p.setPen(QColor(220, 220, 220));
    p.setFont(QFont("Microsoft YaHei", 9, QFont::Bold));
    p.drawText(QRectF(8, 4, 200, 20),
               Qt::AlignLeft | Qt::AlignVCenter,
               "📈 训练曲线（实时）");
}
