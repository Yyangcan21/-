#include "annotationwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    AnnotationWindow w;
    w.show();
    return QApplication::exec();
}
