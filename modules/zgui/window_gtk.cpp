#include "window_gtk.h"

#ifdef USE_GTK2
#include <string>
#include <cstring>
#include <zcore/debug.h>
#include <zcore/util.h>
#include <zmatrix.h>
#include "gtk/gtk.h"

static void destroy();

class zWindow
{
public:
    zWindow() = default;

    std::string name;

    zWindow *prev = nullptr;
    zWindow *next = nullptr;

    GtkWidget * ptr = nullptr;
    GtkWidget * area = nullptr;
    z::Matrix image;
};

static zWindow * window_head = nullptr;

int GUIInitSystem(int argc, char** argv)
{

    static bool initialized = false;

    if(!initialized) {
        gtk_init(&argc, &argv);
        gdk_rgb_init();

        atexit(destroy);

        initialized = true;
    }
    return 0;
}

static int key = 0;
void keyboard_callback(GtkWidget *widget, GdkEventKey *event, gpointer user_data)
{
    key = event->keyval;
    gtk_main_quit();
}


zWindow * findWindow(const std::string& name)
{
    auto window = window_head;
    for(; window != nullptr && name != window->name; window = window->next);

    return window;
}

int zNamedWindow(const char* name, int flags)
{
    GUIInitSystem(0, 0);

    auto window = findWindow(name);
    if(window)
        return 0;

    window = new zWindow();
    window->name = name;

    window->ptr = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window->ptr), name);

    gtk_signal_connect(GTK_OBJECT(window->ptr), "key-press-event",
                       GTK_SIGNAL_FUNC(keyboard_callback), NULL);

    window->area = gtk_drawing_area_new();
    gtk_container_add(GTK_CONTAINER(window->ptr), window->area);

    window->next = window_head;
    window->prev = nullptr;
    if(window_head)
        window_head->prev = window;
    window_head = window;

    return 1;
}

gboolean on_darea_expose (GtkWidget *widget, GdkEventExpose *event, gpointer user_data)
{
    auto image = reinterpret_cast<z::Matrix *>(user_data);
    if(image->channels() == 1){
        gdk_draw_gray_image(widget->window,
                            widget->style->fg_gc[GTK_STATE_NORMAL],
                            0, 0, image->cols, image->rows,
                            GDK_RGB_DITHER_NONE, image->data, image->cols);
    }
    else {
        gdk_draw_rgb_image (widget->window,
                            widget->style->fg_gc[GTK_STATE_NORMAL],
                            0, 0, image->cols, image->rows,
                            GDK_RGB_DITHER_MAX, image->data, image->cols * 3);
    }
    return true;
}

void zShowImage(const char* name, void* arr)
{
    if(!name){
        Z_Error("name can not be null");
    }

    auto window = findWindow(name);

    if(!window) {
        zNamedWindow(name, 0);
        window = findWindow(name);
    }

    if(!window) {
        Z_Error("window is nullptr");
        return;
    }

    auto original = reinterpret_cast<z::Matrix *>(arr);
    if(original->channels() == 3) {
        z::cvtColor(*original, window->image, z::BGR2RGB);
    }
    else {
        window->image = original->clone();
    }

    gtk_drawing_area_size(GTK_DRAWING_AREA(window->area), window->image.cols, window->image.rows);


    gtk_signal_connect(GTK_OBJECT(window->area), "expose-event",
                       GTK_SIGNAL_FUNC(on_darea_expose), &(window->image));

    gtk_widget_show_all(window->ptr);
}

static void destroy()
{

    for(zWindow * window = window_head, *next = nullptr; window != nullptr; window = next) {
        next = window->next;
        delete(window->prev);
    }
}


int zWaitKey(int delay)
{
    if(delay != 0) {
        g_timeout_add(static_cast<guint>(delay), (GSourceFunc)gtk_main_quit, nullptr);
    }

    gtk_main();

    return key;
}

#endif // !USE_GTK2