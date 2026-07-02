#ifndef LOSSCURVEVIEW_H
#define LOSSCURVEVIEW_H

#include <QWidget>
#include <QVector>
#include <QPointF>

/**
 * LossCurveView - 纯 QPainter 自绘的训练曲线控件
 *
 * 功能：
 *   1. 实时绘制 cls_loss / dfl_loss 2 条曲线
 *   2. 自动归一化 Y 轴 + X 轴
 *   3. 图例、网格、坐标轴、悬浮提示全部自绘
 *   4. 零外部依赖（不依赖 QCustomPlot / QChart）
 *
 * 用法：
 *   LossCurveView *cv = new LossCurveView(this);
 *   cv->addDataPoint(epoch, clsLoss, dflLoss, mAP, precision, recall);
 *   cv->clearCurves();
 */
class LossCurveView : public QWidget
{
    Q_OBJECT
public:
    explicit LossCurveView(QWidget *parent = nullptr);

    // 添加一组数据（一个 epoch 一次调用）
    // epoch 从 1 开始；如果某些指标未知，传 -1 表示不画
    void addDataPoint(int epoch,
                      double clsLoss,
                      double dflLoss,
                      double map50,
                      double precision,
                      double recall);

    // 清空所有曲线
    void clearCurves();

    // 设置 X 轴最大值（总 epoch 数），影响 X 轴刻度
    void setMaxEpoch(int n) { m_maxEpoch = qMax(1, n); update(); }

    QSize sizeHint() const override { return QSize(600, 200); }

protected:
    void paintEvent(QPaintEvent *event) override;

private:
    struct Curve {
        QString name;
        QColor  color;
        QVector<QPointF> points;  // (epoch, value)
    };

    QVector<Curve> m_curves;  // 2 条曲线：cls/dfl
    int m_maxEpoch = 50;       // X 轴上限
};

#endif // LOSSCURVEVIEW_H
