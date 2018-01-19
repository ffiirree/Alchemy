#include "gui/gui.h"
#include "window_win32.h"

#if defined _WIN32

#include "zcore/debug.h"
#include <map>
#include <algorithm>
#include <WinUser.h>
#include <Windowsx.h>
#include <commctrl.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

enum
{
    //These 3 flags are used by cvSet/GetWindowProperty
    Z_WND_PROP_FULLSCREEN = 0, //to change/get window's fullscreen property
    Z_WND_PROP_AUTOSIZE = 1, //to change/get window's autosize property
    Z_WND_PROP_ASPECTRATIO = 2, //to change/get window's aspectratio property
    Z_WND_PROP_OPENGL = 3, //to change/get window's opengl support
    Z_WND_PROP_VISIBLE = 4,

    //These 2 flags are used by cvNamedWindow and cvSet/GetWindowProperty
    Z_WINDOW_NORMAL = 0x00000000, //the user can resize the window (no constraint)  / also use to switch a fullscreen window to a normal size
    Z_WINDOW_AUTOSIZE = 0x00000001, //the user cannot resize the window, the size is constrainted by the image displayed
    Z_WINDOW_OPENGL = 0x00001000, //window with opengl support

    //These 3 flags are used by cvNamedWindow and cvSet/GetWindowProperty
    Z_WINDOW_FULLSCREEN = 1,//change the window to fullscreen
    Z_WINDOW_FREERATIO = 0x00000100,//the image expends as much as it can (no ratio constraint)
    Z_WINDOW_KEEPRATIO = 0x00000000//the ration image is respected.
};

static const char* trackbar_text =
"                                                                                             ";


#ifndef WM_MOUSEHWHEEL
#define WM_MOUSEHWHEEL 0x020E
#endif

static void FillBitmapInfo(BITMAPINFO* bmi, int width, int height, int bpp, int origin)
{
    assert(bmi && width >= 0 && height >= 0 && (bpp == 8 || bpp == 24 || bpp == 32));

    auto bmih = &(bmi->bmiHeader);

    memset(bmih, 0, sizeof(*bmih));
    bmih->biSize = sizeof(BITMAPINFOHEADER);
    bmih->biWidth = width;
    bmih->biHeight = origin ? abs(height) : -abs(height);
    bmih->biPlanes = 1;
    bmih->biBitCount = static_cast<unsigned short>(bpp);
    bmih->biCompression = BI_RGB;

    if (bpp == 8)
    {
        auto palette = bmi->bmiColors;
        for (auto i = 0; i < 256; i++)
        {
            palette[i].rgbBlue = palette[i].rgbGreen = palette[i].rgbRed = static_cast<BYTE>(i);
            palette[i].rgbReserved = 0;
        }
    }
}

struct zWindow;

typedef struct zTrackbar
{
    int signature;
    HWND hwnd;
    char* name;
    zTrackbar* next;
    zWindow* parent;
    HWND buddy;
    int* data;
    int pos;
    int maxval;
    int minval;
    void(*notify)(int);
    void(*notify2)(int, void*);
    void* userdata;
    int id;
}zTrackbar;


typedef struct zWindow
{
    int signature;
    HWND hwnd;
    char* name;
    zWindow* prev;
    zWindow* next;
    HWND frame;

    HDC dc;
    HGDIOBJ image;
    int last_key;
    int flags;
    int status;//0 normal, 1 fullscreen (YV)

    zMouseCallback on_mouse;
    void* on_mouse_param;

    struct
    {
        HWND toolbar;
        int pos;
        int rows;
        WNDPROC toolBarProc;
        zTrackbar* first;
    }
    toolbar;

    int width;
    int height;
}
zWindow;

#define HG_BUDDY_WIDTH  130

#ifndef TBIF_SIZE
#define TBIF_SIZE  0x40
#endif

#ifndef TB_SETBUTTONINFO
#define TB_SETBUTTONINFO (WM_USER + 66)
#endif


static LRESULT CALLBACK HighGUIProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
static LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
static LRESULT CALLBACK MainWindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
static void zUpdateWindowPos(zWindow* window);
void zDestroyAllWindows();

static zWindow* hg_windows = 0;

typedef int (* zWin32WindowCallback)(HWND, UINT, WPARAM, LPARAM, int*);
static zWin32WindowCallback hg_on_preprocess = 0, hg_on_postprocess = 0;
static HINSTANCE hg_hinstance = 0;

static const char* highGUIclassName = "HighGUI class";
static const char* mainHighGUIclassName = "Main HighGUI class";

static void zCleanupHighgui()
{
    zDestroyAllWindows();
    UnregisterClass(highGUIclassName, hg_hinstance);
    UnregisterClass(mainHighGUIclassName, hg_hinstance);
}

int GUIInitSystem(int, char**)
{
    static bool wasInitialized = false;

    // check initialization status
    if (!wasInitialized)
    {
        // Initialize the stogare
        hg_windows = 0;

        // Register the class
        WNDCLASS wndc;
        wndc.style = CS_OWNDC | CS_VREDRAW | CS_HREDRAW | CS_DBLCLKS;
        wndc.lpfnWndProc = WindowProc;
        wndc.cbClsExtra = 0;
        wndc.cbWndExtra = 0;
        wndc.hInstance = hg_hinstance;
        wndc.lpszClassName = highGUIclassName;
        wndc.lpszMenuName = highGUIclassName;
        wndc.hIcon = LoadIcon(0, IDI_APPLICATION);
        wndc.hCursor = LoadCursor(0, IDC_CROSS);
        wndc.hbrBackground = (HBRUSH)GetStockObject(GRAY_BRUSH);

        RegisterClass(&wndc);

        wndc.lpszClassName = mainHighGUIclassName;
        wndc.lpszMenuName = mainHighGUIclassName;
        wndc.hbrBackground = (HBRUSH)GetStockObject(GRAY_BRUSH);
        wndc.lpfnWndProc = MainWindowProc;

        RegisterClass(&wndc);
        atexit(zCleanupHighgui);

        wasInitialized = true;
    }

    setlocale(LC_NUMERIC, "C");

    return 0;
}


static zWindow* zFindWindowByName(const char* name)
{
    auto window = hg_windows;

    for (; window != 0 && strcmp(name, window->name) != 0; window = window->next)
        ;

    return window;
}


static zWindow* zWindowByHWND(HWND hwnd)
{
    auto window = reinterpret_cast<zWindow*>(GetWindowLong(hwnd, GWL_USERDATA));
    return window != 0 && hg_windows != 0 &&
        window->signature == Z_WINDOW_MAGIC_VAL ? window : 0;
}


static zTrackbar* izTrackbarByHWND(HWND hwnd)
{
    auto trackbar = reinterpret_cast<zTrackbar*>(GetWindowLong(hwnd, GWL_USERDATA));
    return trackbar != 0 && trackbar->signature == Z_TRACKBAR_MAGIC_VAL &&
        trackbar->hwnd == hwnd ? trackbar : 0;
}


static const char* windowPosRootKey = "Software\\OpenCV\\HighGUI\\Windows\\";

// Window positions saving/loading added by Philip Gruebele.
//<a href="mailto:pgruebele@cox.net">pgruebele@cox.net</a>
// Restores the window position from the registry saved position.
static void loadWindowPos(const char* name, alchemy::Rect& rect)
{
    HKEY hkey;
    char szKey[1024];
    strcpy(szKey, windowPosRootKey);
    strcat(szKey, name);

    rect.x = rect.y = CW_USEDEFAULT;
    rect.width = rect.height = 320;

    if (RegOpenKeyEx(HKEY_CURRENT_USER, szKey, 0, KEY_QUERY_VALUE, &hkey) == ERROR_SUCCESS)
    {
        // Yes we are installed.
        DWORD dwType = 0;
        DWORD dwSize = sizeof(int);

        RegQueryValueEx(hkey, "Left", NULL, &dwType, (BYTE*)&rect.x, &dwSize);
        RegQueryValueEx(hkey, "Top", NULL, &dwType, (BYTE*)&rect.y, &dwSize);
        RegQueryValueEx(hkey, "Width", NULL, &dwType, (BYTE*)&rect.width, &dwSize);
        RegQueryValueEx(hkey, "Height", NULL, &dwType, (BYTE*)&rect.height, &dwSize);

        if (rect.x != (int)CW_USEDEFAULT && (rect.x < -200 || rect.x > 3000))
            rect.x = 100;
        if (rect.y != (int)CW_USEDEFAULT && (rect.y < -200 || rect.y > 3000))
            rect.y = 100;

        if (rect.width != (int)CW_USEDEFAULT && (rect.width < 0 || rect.width > 3000))
            rect.width = 100;
        if (rect.height != (int)CW_USEDEFAULT && (rect.height < 0 || rect.height > 3000))
            rect.height = 100;

        RegCloseKey(hkey);
    }
}


// Window positions saving/loading added by Philip Gruebele.
//<a href="mailto:pgruebele@cox.net">pgruebele@cox.net</a>
// philipg.  Saves the window position in the registry
static void saveWindowPos(const char* name, alchemy::Rect rect)
{
    static const DWORD MAX_RECORD_COUNT = 100;
    HKEY hkey;
    char szKey[1024];
    char rootKey[1024];
    strcpy(szKey, windowPosRootKey);
    strcat(szKey, name);

    if (RegOpenKeyEx(HKEY_CURRENT_USER, szKey, 0, KEY_READ, &hkey) != ERROR_SUCCESS)
    {
        HKEY hroot;
        DWORD count = 0;
        FILETIME oldestTime = { UINT_MAX, UINT_MAX };
        char oldestKey[1024];
        char currentKey[1024];

        strcpy(rootKey, windowPosRootKey);
        rootKey[strlen(rootKey) - 1] = '\0';
        if (RegCreateKeyEx(HKEY_CURRENT_USER, rootKey, 0, nullptr, REG_OPTION_NON_VOLATILE, KEY_READ + KEY_WRITE, 0, &hroot, nullptr) != ERROR_SUCCESS)
            //RegOpenKeyEx( HKEY_CURRENT_USER,rootKey,0,KEY_READ,&hroot) != ERROR_SUCCESS )
            return;

        for (;;)
        {
            DWORD csize = sizeof(currentKey);
            FILETIME accesstime = { 0, 0 };
            LONG code = RegEnumKeyEx(hroot, count, currentKey, &csize, nullptr, nullptr, nullptr, &accesstime);
            if (code != ERROR_SUCCESS && code != ERROR_MORE_DATA)
                break;
            count++;
            if (oldestTime.dwHighDateTime > accesstime.dwHighDateTime ||
                (oldestTime.dwHighDateTime == accesstime.dwHighDateTime &&
                    oldestTime.dwLowDateTime > accesstime.dwLowDateTime))
            {
                oldestTime = accesstime;
                strcpy(oldestKey, currentKey);
            }
        }

        if (count >= MAX_RECORD_COUNT)
            RegDeleteKey(hroot, oldestKey);
        RegCloseKey(hroot);

        if (RegCreateKeyEx(HKEY_CURRENT_USER, szKey, 0, nullptr, REG_OPTION_NON_VOLATILE, KEY_WRITE, 0, &hkey, NULL) != ERROR_SUCCESS)
            return;
    }
    else
    {
        RegCloseKey(hkey);
        if (RegOpenKeyEx(HKEY_CURRENT_USER, szKey, 0, KEY_WRITE, &hkey) != ERROR_SUCCESS)
            return;
    }

    RegSetValueEx(hkey, "Left", 0, REG_DWORD, (BYTE*)&rect.x, sizeof(rect.x));
    RegSetValueEx(hkey, "Top", 0, REG_DWORD, (BYTE*)&rect.y, sizeof(rect.y));
    RegSetValueEx(hkey, "Width", 0, REG_DWORD, (BYTE*)&rect.width, sizeof(rect.width));
    RegSetValueEx(hkey, "Height", 0, REG_DWORD, (BYTE*)&rect.height, sizeof(rect.height));
    RegCloseKey(hkey);
}



int zNamedWindow(const char* name, int flags)
{
    int result = 0;

    HWND mainhWnd;
    DWORD defStyle = WS_VISIBLE | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU;
    alchemy::Rect rect;

    GUIInitSystem(0, 0);

	assert(name);

    // Check the name in the storage
    auto window = zFindWindowByName(name);
    if (window != 0)
    {
        result = 1;
        return 0;
    }

    if (!(flags & Z_WINDOW_AUTOSIZE))//YV add border in order to resize the window
        defStyle |= WS_SIZEBOX;


    loadWindowPos(name, rect);

    mainhWnd = CreateWindow(
        "Main HighGUI class", 
        name, 
        defStyle | WS_OVERLAPPED,
        rect.x, rect.y, 
        rect.width, rect.height,
        0, 0, hg_hinstance, 0);

	assert(mainhWnd);

    ShowWindow(mainhWnd, SW_SHOW);

    //YV- remove one border by changing the style
    auto hWnd = CreateWindow(
        "HighGUI class",
        "",
        (defStyle & ~WS_SIZEBOX) | WS_CHILD,
        CW_USEDEFAULT, 0, 
        rect.width, rect.height, 
        mainhWnd, 0, hg_hinstance, 0);

	assert(hWnd);

    ShowWindow(hWnd, SW_SHOW);

    auto len = static_cast<int>(strlen(name));
    window = reinterpret_cast<zWindow *>(malloc(sizeof(zWindow) + len + 1));

    window->signature = Z_WINDOW_MAGIC_VAL;
    window->hwnd = hWnd;
    window->frame = mainhWnd;
    window->name = reinterpret_cast<char*>(window + 1);
    memcpy(window->name, name, len + 1);
    window->flags = flags;
    window->image = nullptr;


    window->dc = CreateCompatibleDC(nullptr);

    window->last_key = 0;
    window->status = Z_WINDOW_NORMAL;//YV

    window->on_mouse = 0;
    window->on_mouse_param = 0;

    memset(&window->toolbar, 0, sizeof(window->toolbar));

    window->next = hg_windows;
    window->prev = 0;
    if (hg_windows)
        hg_windows->prev = window;
    hg_windows = window;
    SetWindowLong(hWnd, GWL_USERDATA, reinterpret_cast<size_t>(window));
    SetWindowLong(mainhWnd, GWL_USERDATA, reinterpret_cast<size_t>(window));

    // Recalculate window pos
    zUpdateWindowPos(window);

    result = 1;

    return result;
}


static void zRemoveWindow(zWindow* window)
{
    zTrackbar* trackbar = NULL;
    RECT wrect = { 0,0,0,0 };

    if (window->frame)
        GetWindowRect(window->frame, &wrect);
    if (window->name)
        saveWindowPos(window->name, alchemy::Rect(wrect.left, wrect.top,
            wrect.right - wrect.left, wrect.bottom - wrect.top));

    if (window->hwnd)
        SetWindowLong(window->hwnd, GWL_USERDATA, 0);
    if (window->frame)
        SetWindowLong(window->frame, GWL_USERDATA, 0);

    if (window->toolbar.toolbar)
        SetWindowLong(window->toolbar.toolbar, GWL_USERDATA, 0);

    if (window->prev)
        window->prev->next = window->next;
    else
        hg_windows = window->next;

    if (window->next)
        window->next->prev = window->prev;

    window->prev = window->next = 0;

    if (window->dc && window->image)
        DeleteObject(SelectObject(window->dc, window->image));

    if (window->dc)
        DeleteDC(window->dc);

    for (trackbar = window->toolbar.first; trackbar != 0; )
    {
        zTrackbar* next = trackbar->next;
        if (trackbar->hwnd)
        {
            SetWindowLong(trackbar->hwnd, GWL_USERDATA, 0);
            free(&trackbar);
        }
        trackbar = next;
    }

    free(window);
}


void zDestroyWindow(const char* name)
{
	assert(name);

    auto window = zFindWindowByName(name);
    if (!window)
        return;

    auto mainhWnd = window->frame;

    SendMessage(window->hwnd, WM_CLOSE, 0, 0);
    SendMessage(mainhWnd, WM_CLOSE, 0, 0);
}


static void zScreenToClient(HWND hwnd, RECT* rect)
{
    POINT p;
    p.x = rect->left;
    p.y = rect->top;
    ScreenToClient(hwnd, &p);
    OffsetRect(rect, p.x - rect->left, p.y - rect->top);
}


/* Calculatess the window coordinates relative to the upper left corner of the mainhWnd window */
static RECT calcWindowRect(zWindow* window)
{
    const auto gutter = 1;
    RECT crect, trect, rect;

    assert(window);

    GetClientRect(window->frame, &crect);
    if (window->toolbar.toolbar)
    {
        GetWindowRect(window->toolbar.toolbar, &trect);
        zScreenToClient(window->frame, &trect);
        SubtractRect(&rect, &crect, &trect);
    }
    else
        rect = crect;

    rect.top += gutter;
    rect.left += gutter;
    rect.bottom -= gutter;
    rect.right -= gutter;

    return rect;
}

// returns TRUE if there is a problem such as ERROR_IO_PENDING.
static bool zGetBitmapData(zWindow* window, SIZE* size, int* channels, void** data)
{
    BITMAP bmp;
    GdiFlush();
    HGDIOBJ h = GetCurrentObject(window->dc, OBJ_BITMAP);
    if (size)
        size->cx = size->cy = 0;
    if (data)
        *data = 0;

    if (h == NULL)
        return true;
    if (GetObject(h, sizeof(bmp), &bmp) == 0)
        return true;

    if (size)
    {
        size->cx = abs(bmp.bmWidth);
        size->cy = abs(bmp.bmHeight);
    }

    if (channels)
        *channels = bmp.bmBitsPixel / 8;

    if (data)
        *data = bmp.bmBits;

    return false;
}


static void zUpdateWindowPos(zWindow* window)
{
    RECT rect;
    assert(window);

    if ((window->flags & Z_WINDOW_AUTOSIZE) && window->image)
    {
        int i;
        SIZE size = { 0,0 };
        zGetBitmapData(window, &size, 0, 0);

        // Repeat two times because after the first resizing of the mainhWnd window
        // toolbar may resize too
        for (i = 0; i < (window->toolbar.toolbar ? 2 : 1); i++)
        {
            RECT rmw, rw = calcWindowRect(window);
            MoveWindow(window->hwnd, rw.left, rw.top,
                rw.right - rw.left + 1, rw.bottom - rw.top + 1, FALSE);
            GetClientRect(window->hwnd, &rw);
            GetWindowRect(window->frame, &rmw);
            // Resize the mainhWnd window in order to make the bitmap fit into the child window
            MoveWindow(window->frame, rmw.left, rmw.top,
                rmw.right - rmw.left + size.cx - rw.right + rw.left,
                rmw.bottom - rmw.top + size.cy - rw.bottom + rw.top, TRUE);
        }
    }

    rect = calcWindowRect(window);
    MoveWindow(window->hwnd, rect.left, rect.top,
        rect.right - rect.left + 1,
        rect.bottom - rect.top + 1, TRUE);
}

void cpy(alchemy::Matrix8u &src, char * arr)
{
    auto step = (src.cols * src.channels() + 3) & -4;
	for(auto i = 0; i < src.rows; ++i) {
		for(auto j = 0; j < src.cols; ++j) {
			for(auto k = 0; k < src.channels(); ++k) {
				arr[i * step + j * src.channels() + k] = src.at(i, j, k);
			}
		}
	}
}

void zShowImage(const char* name, const void* arr)
{
	assert(name);

	SIZE size = { 0, 0 };
    int channels = 0;
    void* dst_ptr = nullptr;
    alchemy::Matrix dst;
    bool changed_size = false; // philipg

	auto window = zFindWindowByName(name);
    if (!window) {
        zNamedWindow(name, Z_WINDOW_AUTOSIZE);
        window = zFindWindowByName(name);
    }

    if (!window || !arr)
        return; // keep silence here.

    auto image = reinterpret_cast<const alchemy::Matrix *>(arr);

    if (window->image)
        // if there is something wrong with these system calls, we cannot display image...
        if (zGetBitmapData(window, &size, &channels, &dst_ptr))
            return;

    if (size.cx != image->cols || size.cy != image->rows || channels != image->channels())
    {
        changed_size = true;

        uint8_t buffer[sizeof(BITMAPINFO) + 255 * sizeof(RGBQUAD)];
        auto binfo = reinterpret_cast<BITMAPINFO*>(buffer);

        DeleteObject(SelectObject(window->dc, window->image));
        window->image = 0;

        size.cx = image->cols;
        size.cy = image->rows;
        channels = image->channels();

        FillBitmapInfo(binfo, size.cx, size.cy, channels * 8, 1);

        window->image = SelectObject(window->dc, CreateDIBSection(window->dc, binfo,
            DIB_RGB_COLORS, &dst_ptr, 0, 0));
    }

    convertImage(image, &dst);
    cpy(dst, reinterpret_cast<char *>(dst_ptr));

    // ony resize window if needed
    if (changed_size)
        zUpdateWindowPos(window);
    InvalidateRect(window->hwnd, 0, 0);
}


void resizeWindow(const char* name, int width, int height)
{
	assert(name);

    RECT rmw, rw;

    auto window = zFindWindowByName(name);
    if (!window)
        return;

    // Repeat two times because after the first resizing of the mainhWnd window
    // toolbar may resize too
    for (int i = 0; i < (window->toolbar.toolbar ? 2 : 1); i++)
    {
        rw = calcWindowRect(window);
        MoveWindow(window->hwnd, rw.left, rw.top,
            rw.right - rw.left + 1, rw.bottom - rw.top + 1, FALSE);
        GetClientRect(window->hwnd, &rw);
        GetWindowRect(window->frame, &rmw);
        // Resize the mainhWnd window in order to make the bitmap fit into the child window
        MoveWindow(window->frame, rmw.left, rmw.top,
            rmw.right - rmw.left + width - rw.right + rw.left,
            rmw.bottom - rmw.top + height - rw.bottom + rw.top, TRUE);
    }

    RECT rect = calcWindowRect(window);
    MoveWindow(window->hwnd, rect.left, rect.top,
        rect.right - rect.left + 1, rect.bottom - rect.top + 1, TRUE);
}


void zMoveWindow(const char* name, int x, int y)
{
	assert(name);

    RECT rect;

    zWindow * window = zFindWindowByName(name);
    if (!window)
        return;

    GetWindowRect(window->frame, &rect);
    MoveWindow(window->frame, x, y, rect.right - rect.left, rect.bottom - rect.top, TRUE);
}

enum
{
    Z_EVENT_FLAG_LBUTTON = 1,
    Z_EVENT_FLAG_RBUTTON = 2,
    Z_EVENT_FLAG_MBUTTON = 4,
    Z_EVENT_FLAG_CTRLKEY = 8,
    Z_EVENT_FLAG_SHIFTKEY = 16,
    Z_EVENT_FLAG_ALTKEY = 32
};

enum
{
    Z_EVENT_MOUSEMOVE = 0,
    Z_EVENT_LBUTTONDOWN = 1,
    Z_EVENT_RBUTTONDOWN = 2,
    Z_EVENT_MBUTTONDOWN = 3,
    Z_EVENT_LBUTTONUP = 4,
    Z_EVENT_RBUTTONUP = 5,
    Z_EVENT_MBUTTONUP = 6,
    Z_EVENT_LBUTTONDBLCLK = 7,
    Z_EVENT_RBUTTONDBLCLK = 8,
    Z_EVENT_MBUTTONDBLCLK = 9,
    Z_EVENT_MOUSEWHEEL = 10,
    Z_EVENT_MOUSEHWHEEL = 11
};
static LRESULT CALLBACK MainWindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    auto window = zWindowByHWND(hwnd);
    if (!window)
        return DefWindowProc(hwnd, uMsg, wParam, lParam);

    switch (uMsg)
    {
    case WM_COPY:
        ::SendMessage(window->hwnd, uMsg, wParam, lParam);
        break;

    case WM_DESTROY:

        zRemoveWindow(window);
        // Do nothing!!!
        //PostQuitMessage(0);
        break;

    case WM_GETMINMAXINFO:
        if (!(window->flags & Z_WINDOW_AUTOSIZE))
        {
            MINMAXINFO* minmax = (MINMAXINFO*)lParam;
            RECT rect;
            LRESULT retval = DefWindowProc(hwnd, uMsg, wParam, lParam);

            minmax->ptMinTrackSize.y = 100;
            minmax->ptMinTrackSize.x = 100;

            if (window->toolbar.first)
            {
                GetWindowRect(window->toolbar.first->hwnd, &rect);
                minmax->ptMinTrackSize.y += window->toolbar.rows*(rect.bottom - rect.top);
                minmax->ptMinTrackSize.x = std::max(rect.right - rect.left + HG_BUDDY_WIDTH, static_cast<long>(HG_BUDDY_WIDTH * 2));
            }
            return retval;
        }
        break;

    case WM_WINDOWPOSCHANGED:
    {
        WINDOWPOS* pos = (WINDOWPOS*)lParam;

        // Update the toolbar pos/size
        if (window->toolbar.toolbar)
        {
            RECT rect;
            GetWindowRect(window->toolbar.toolbar, &rect);
            MoveWindow(window->toolbar.toolbar, 0, 0, pos->cx, rect.bottom - rect.top, TRUE);
        }

        if (!(window->flags & Z_WINDOW_AUTOSIZE))
            zUpdateWindowPos(window);

        break;
    }

    case WM_WINDOWPOSCHANGING:
    {
        // Snap window to screen edges with multi-monitor support. // Adi Shavit
        LPWINDOWPOS pos = (LPWINDOWPOS)lParam;

        RECT rect;
        GetWindowRect(window->frame, &rect);

        HMONITOR hMonitor;
        hMonitor = MonitorFromRect(&rect, MONITOR_DEFAULTTONEAREST);

        MONITORINFO mi;
        mi.cbSize = sizeof(mi);
        GetMonitorInfo(hMonitor, &mi);

        const int SNAP_DISTANCE = 15;

        if (abs(pos->x - mi.rcMonitor.left) <= SNAP_DISTANCE)
            pos->x = mi.rcMonitor.left;               // snap to left edge
        else
            if (abs(pos->x + pos->cx - mi.rcMonitor.right) <= SNAP_DISTANCE)
                pos->x = mi.rcMonitor.right - pos->cx; // snap to right edge

        if (abs(pos->y - mi.rcMonitor.top) <= SNAP_DISTANCE)
            pos->y = mi.rcMonitor.top;                 // snap to top edge
        else
            if (abs(pos->y + pos->cy - mi.rcMonitor.bottom) <= SNAP_DISTANCE)
                pos->y = mi.rcMonitor.bottom - pos->cy; // snap to bottom edge
    }

    case WM_ACTIVATE:
        if (LOWORD(wParam) == WA_ACTIVE || LOWORD(wParam) == WA_CLICKACTIVE)
            SetFocus(window->hwnd);
        break;

    case WM_MOUSEWHEEL:
    case WM_MOUSEHWHEEL:
        if (window->on_mouse)
        {
            int flags = (wParam & MK_LBUTTON ? Z_EVENT_FLAG_LBUTTON : 0) |
                (wParam & MK_RBUTTON ? Z_EVENT_FLAG_RBUTTON : 0) |
                (wParam & MK_MBUTTON ? Z_EVENT_FLAG_MBUTTON : 0) |
                (wParam & MK_CONTROL ? Z_EVENT_FLAG_CTRLKEY : 0) |
                (wParam & MK_SHIFT ? Z_EVENT_FLAG_SHIFTKEY : 0) |
                (GetKeyState(VK_MENU) < 0 ? Z_EVENT_FLAG_ALTKEY : 0);
            int event = (uMsg == WM_MOUSEWHEEL ? Z_EVENT_MOUSEWHEEL : Z_EVENT_MOUSEHWHEEL);

            // Set the wheel delta of mouse wheel to be in the upper word of 'event'
            int delta = GET_WHEEL_DELTA_WPARAM(wParam);
            flags |= (delta << 16);

            POINT pt;
            pt.x = GET_X_LPARAM(lParam);
            pt.y = GET_Y_LPARAM(lParam);
            ::ScreenToClient(hwnd, &pt); // Convert screen coordinates to client coordinates.

            RECT rect;
            GetClientRect(window->hwnd, &rect);

            SIZE size = { 0,0 };
            zGetBitmapData(window, &size, 0, 0);

            window->on_mouse(event, pt.x*size.cx / std::max(rect.right - rect.left, 1l),
                pt.y*size.cy / std::max(rect.bottom - rect.top, 1l), flags,
                window->on_mouse_param);
        }
        break;

    case WM_ERASEBKGND:
    {
        RECT cr, tr, wrc;
        HRGN rgn, rgn1, rgn2;
        int ret;
        HDC hdc = (HDC)wParam;
        GetWindowRect(window->hwnd, &cr);
        zScreenToClient(window->frame, &cr);
        if (window->toolbar.toolbar)
        {
            GetWindowRect(window->toolbar.toolbar, &tr);
            zScreenToClient(window->frame, &tr);
        }
        else
            tr.left = tr.top = tr.right = tr.bottom = 0;

        GetClientRect(window->frame, &wrc);

        rgn = CreateRectRgn(0, 0, wrc.right, wrc.bottom);
        rgn1 = CreateRectRgn(cr.left, cr.top, cr.right, cr.bottom);
        rgn2 = CreateRectRgn(tr.left, tr.top, tr.right, tr.bottom);
        ret = CombineRgn(rgn, rgn, rgn1, RGN_DIFF);
        ret = CombineRgn(rgn, rgn, rgn2, RGN_DIFF);

        if (ret != NULLREGION && ret != ERROR)
            FillRgn(hdc, rgn, (HBRUSH)GetClassLong(hwnd, GCL_HBRBACKGROUND));

        DeleteObject(rgn);
        DeleteObject(rgn1);
        DeleteObject(rgn2);
    }
    return 1;
    }

    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}


static LRESULT CALLBACK HighGUIProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    auto window = zWindowByHWND(hwnd);
    if (!window)
        // This window is not mentioned in HighGUI storage
        // Actually, this should be error except for the case of calls to CreateWindow
        return DefWindowProc(hwnd, uMsg, wParam, lParam);

    // Process the message
    switch (uMsg)
    {
    case WM_COPY:
    {
        if (!::OpenClipboard(hwnd))
            break;

        HDC hDC = 0;
        HDC memDC = 0;
        HBITMAP memBM = 0;

        // We'll use a do-while(0){} scope as a single-run breakable scope
        // Upon any error we can jump out of the single-time while scope to clean up the resources.
        do
        {
            if (!::EmptyClipboard())
                break;

            if (!window->image)
                break;

            // Get window device context
            if (0 == (hDC = ::GetDC(hwnd)))
                break;

            // Create another DC compatible with hDC
            if (0 == (memDC = ::CreateCompatibleDC(hDC)))
                break;

            // Determine the bitmap's dimensions
            int nchannels = 3;
            SIZE size = { 0,0 };
            zGetBitmapData(window, &size, &nchannels, 0);

            // Create bitmap to draw on and it in the new DC
            if (0 == (memBM = ::CreateCompatibleBitmap(hDC, size.cx, size.cy)))
                break;

            if (!::SelectObject(memDC, memBM))
                break;

            // Begin drawing to DC
            if (!::SetStretchBltMode(memDC, COLORONCOLOR))
                break;

            RGBQUAD table[256];
            if (1 == nchannels)
            {
                for (int i = 0; i < 256; ++i)
                {
                    table[i].rgbBlue = (unsigned char)i;
                    table[i].rgbGreen = (unsigned char)i;
                    table[i].rgbRed = (unsigned char)i;
                }
                if (!::SetDIBColorTable(window->dc, 0, 255, table))
                    break;
            }

            // The image copied to the clipboard will be in its original size, regardless if the window itself was resized.

            // Render the image to the dc/bitmap (at original size).
            if (!::BitBlt(memDC, 0, 0, size.cx, size.cy, window->dc, 0, 0, SRCCOPY))
                break;

            // Finally, set bitmap to clipboard
            ::SetClipboardData(CF_BITMAP, memBM);
        } while (0, 0); // (0,0) instead of (0) to avoid MSVC compiler warning C4127: "conditional expression is constant"

                        //////////////////////////////////////////////////////////////////////////
                        // if handle is allocated (i.e. != 0) then clean-up.
        if (memBM) ::DeleteObject(memBM);
        if (memDC) ::DeleteDC(memDC);
        if (hDC)   ::ReleaseDC(hwnd, hDC);
        ::CloseClipboard();
        break;
    }

    case WM_WINDOWPOSCHANGING:
    {
        LPWINDOWPOS pos = (LPWINDOWPOS)lParam;
        RECT rect = calcWindowRect(window);
        pos->x = rect.left;
        pos->y = rect.top;
        pos->cx = rect.right - rect.left + 1;
        pos->cy = rect.bottom - rect.top + 1;
    }
    break;

    case WM_LBUTTONDOWN:
    case WM_RBUTTONDOWN:
    case WM_MBUTTONDOWN:
    case WM_LBUTTONDBLCLK:
    case WM_RBUTTONDBLCLK:
    case WM_MBUTTONDBLCLK:
    case WM_LBUTTONUP:
    case WM_RBUTTONUP:
    case WM_MBUTTONUP:
    case WM_MOUSEMOVE:
        if (window->on_mouse)
        {
            POINT pt;

            int flags = (wParam & MK_LBUTTON ? Z_EVENT_FLAG_LBUTTON : 0) |
                (wParam & MK_RBUTTON ? Z_EVENT_FLAG_RBUTTON : 0) |
                (wParam & MK_MBUTTON ? Z_EVENT_FLAG_MBUTTON : 0) |
                (wParam & MK_CONTROL ? Z_EVENT_FLAG_CTRLKEY : 0) |
                (wParam & MK_SHIFT ? Z_EVENT_FLAG_SHIFTKEY : 0) |
                (GetKeyState(VK_MENU) < 0 ? Z_EVENT_FLAG_ALTKEY : 0);
            int event = uMsg == WM_LBUTTONDOWN ? Z_EVENT_LBUTTONDOWN :
                uMsg == WM_RBUTTONDOWN ? Z_EVENT_RBUTTONDOWN :
                uMsg == WM_MBUTTONDOWN ? Z_EVENT_MBUTTONDOWN :
                uMsg == WM_LBUTTONUP ? Z_EVENT_LBUTTONUP :
                uMsg == WM_RBUTTONUP ? Z_EVENT_RBUTTONUP :
                uMsg == WM_MBUTTONUP ? Z_EVENT_MBUTTONUP :
                uMsg == WM_LBUTTONDBLCLK ? Z_EVENT_LBUTTONDBLCLK :
                uMsg == WM_RBUTTONDBLCLK ? Z_EVENT_RBUTTONDBLCLK :
                uMsg == WM_MBUTTONDBLCLK ? Z_EVENT_MBUTTONDBLCLK :
                Z_EVENT_MOUSEMOVE;
            if (uMsg == WM_LBUTTONDOWN || uMsg == WM_RBUTTONDOWN || uMsg == WM_MBUTTONDOWN)
                SetCapture(hwnd);
            if (uMsg == WM_LBUTTONUP || uMsg == WM_RBUTTONUP || uMsg == WM_MBUTTONUP)
                ReleaseCapture();

            pt.x = GET_X_LPARAM(lParam);
            pt.y = GET_Y_LPARAM(lParam);

            if (window->flags & Z_WINDOW_AUTOSIZE)
            {
                // As user can't change window size, do not scale window coordinates. Underlying windowing system
                // may prevent full window from being displayed and in this case coordinates should not be scaled.
                window->on_mouse(event, pt.x, pt.y, flags, window->on_mouse_param);
            }
            else {
                // Full window is displayed using different size. Scale coordinates to match underlying positions.
                RECT rect;
                SIZE size = { 0, 0 };

                GetClientRect(window->hwnd, &rect);
                zGetBitmapData(window, &size, 0, 0);

                window->on_mouse(event, pt.x*size.cx / std::max(rect.right - rect.left, 1l),
                    pt.y*size.cy / std::max(rect.bottom - rect.top, 1l), flags,
                    window->on_mouse_param);
            }
        }
        break;

    case WM_PAINT:
        if (window->image != 0)
        {
            int nchannels = 3;
            SIZE size = { 0,0 };
            PAINTSTRUCT paint;
            HDC hdc;
            RGBQUAD table[256];

            // Determine the bitmap's dimensions
            zGetBitmapData(window, &size, &nchannels, 0);

            hdc = BeginPaint(hwnd, &paint);
            SetStretchBltMode(hdc, COLORONCOLOR);

            if (nchannels == 1)
            {
                int i;
                for (i = 0; i < 256; i++)
                {
                    table[i].rgbBlue = (unsigned char)i;
                    table[i].rgbGreen = (unsigned char)i;
                    table[i].rgbRed = (unsigned char)i;
                }
                SetDIBColorTable(window->dc, 0, 255, table);
            }

            if (window->flags & Z_WINDOW_AUTOSIZE)
            {
                BitBlt(hdc, 0, 0, size.cx, size.cy, window->dc, 0, 0, SRCCOPY);
            }
            else
            {
                RECT rect;
                GetClientRect(window->hwnd, &rect);
                StretchBlt(hdc, 0, 0, rect.right - rect.left, rect.bottom - rect.top,
                    window->dc, 0, 0, size.cx, size.cy, SRCCOPY);
            }
            //DeleteDC(hdc);
            EndPaint(hwnd, &paint);
        }
        else
        {
            return DefWindowProc(hwnd, uMsg, wParam, lParam);
        }
        return 0;

    case WM_ERASEBKGND:
        if (window->image)
            return 0;
        break;

    case WM_DESTROY:

        zRemoveWindow(window);
        // Do nothing!!!
        //PostQuitMessage(0);
        break;

    case WM_SETCURSOR:
        SetCursor((HCURSOR)GetClassLong(hwnd, GCL_HCURSOR));
        return 0;

    case WM_KEYDOWN:
        window->last_key = (int)wParam;
        return 0;

    case WM_SIZE:
        window->width = LOWORD(lParam);
        window->height = HIWORD(lParam);
    }

    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}


static LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    if (hg_on_preprocess)
    {
        int was_processed = 0;
        int rethg = hg_on_preprocess(hwnd, uMsg, wParam, lParam, &was_processed);
        if (was_processed)
            return rethg;
    }
    LRESULT ret = HighGUIProc(hwnd, uMsg, wParam, lParam);

    if (hg_on_postprocess)
    {
        int was_processed = 0;
        int rethg = hg_on_postprocess(hwnd, uMsg, wParam, lParam, &was_processed);
        if (was_processed)
            return rethg;
    }

    return ret;
}


static void zUpdateTrackbar(zTrackbar* trackbar, int pos)
{
    const int max_name_len = 10;
    const char* suffix = "";
    char pos_text[32];

    if (trackbar->data)
        *trackbar->data = pos;

    if (trackbar->pos != pos)
    {
        trackbar->pos = pos;
        if (trackbar->notify2)
            trackbar->notify2(pos, trackbar->userdata);
        if (trackbar->notify)
            trackbar->notify(pos);

        auto name_len = static_cast<int>(strlen(trackbar->name));

        if (name_len > max_name_len)
        {
            int start_len = max_name_len * 2 / 3;
            int end_len = max_name_len - start_len - 2;
            memcpy(pos_text, trackbar->name, start_len);
            memcpy(pos_text + start_len, "...", 3);
            memcpy(pos_text + start_len + 3, trackbar->name + name_len - end_len, end_len + 1);
        }
        else
        {
            memcpy(pos_text, trackbar->name, name_len + 1);
        }

        sprintf(pos_text + strlen(pos_text), "%s: %d\n", suffix, pos);
        SetWindowText(trackbar->buddy, pos_text);
    }
}


static LRESULT CALLBACK HGToolbarProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    auto window = zWindowByHWND(hwnd);
    if (!window)
        return DefWindowProc(hwnd, uMsg, wParam, lParam);

    // Control messages processing
    switch (uMsg)
    {
        // Slider processing
    case WM_HSCROLL:
    {
        HWND slider = (HWND)lParam;
        int pos = (int)SendMessage(slider, TBM_GETPOS, 0, 0);
        zTrackbar* trackbar = izTrackbarByHWND(slider);

        if (trackbar)
        {
            if (trackbar->pos != pos)
                zUpdateTrackbar(trackbar, pos);
        }

        SetFocus(window->hwnd);
        return 0;
    }

    case WM_NCCALCSIZE:
    {
        LRESULT ret = CallWindowProc(window->toolbar.toolBarProc, hwnd, uMsg, wParam, lParam);
        int rows = (int)SendMessage(hwnd, TB_GETROWS, 0, 0);

        if (window->toolbar.rows != rows)
        {
            SendMessage(window->toolbar.toolbar, TB_BUTTONCOUNT, 0, 0);
            zTrackbar* trackbar = window->toolbar.first;

            for (; trackbar != 0; trackbar = trackbar->next)
            {
                RECT rect;
                SendMessage(window->toolbar.toolbar, TB_GETITEMRECT,
                    (WPARAM)trackbar->id, (LPARAM)&rect);
                MoveWindow(trackbar->hwnd, rect.left + HG_BUDDY_WIDTH, rect.top,
                    rect.right - rect.left - HG_BUDDY_WIDTH,
                    rect.bottom - rect.top, FALSE);
                MoveWindow(trackbar->buddy, rect.left, rect.top,
                    HG_BUDDY_WIDTH, rect.bottom - rect.top, FALSE);
            }
            window->toolbar.rows = rows;
        }
        return ret;
    }
    }

    return CallWindowProc(window->toolbar.toolBarProc, hwnd, uMsg, wParam, lParam);
}


void zDestroyAllWindows()
{
    auto window = hg_windows;

    while (window)
    {
        HWND mainhWnd = window->frame;
        HWND hwnd = window->hwnd;
        window = window->next;

        SendMessage(hwnd, WM_CLOSE, 0, 0);
        SendMessage(mainhWnd, WM_CLOSE, 0, 0);
    }
}

static void showSaveDialog(zWindow* window)
{
    if (!window || !window->image)
        return;

    SIZE sz;
    int channels;
    void* data;
    if (zGetBitmapData(window, &sz, &channels, &data))
        return; // nothing to save

    char szFileName[MAX_PATH] = "";
    // try to use window title as file name
    GetWindowText(window->frame, szFileName, MAX_PATH);

    OPENFILENAME ofn;
    ZeroMemory(&ofn, sizeof(ofn));
#ifdef OPENFILENAME_SIZE_VERSION_400
    // we are not going to use new fields any way
    ofn.lStructSize = OPENFILENAME_SIZE_VERSION_400;
#else
    ofn.lStructSize = sizeof(ofn);
#endif
    ofn.hwndOwner = window->hwnd;
    ofn.lpstrFilter =
#ifdef HAVE_PNG
        "Portable Network Graphics files (*.png)\0*.png\0"
#endif
        "Windows bitmap (*.bmp;*.dib)\0*.bmp;*.dib\0"
#ifdef HAVE_JPEG
        "JPEG files (*.jpeg;*.jpg;*.jpe)\0*.jpeg;*.jpg;*.jpe\0"
#endif
#ifdef HAVE_TIFF
        "TIFF Files (*.tiff;*.tif)\0*.tiff;*.tif\0"
#endif
#ifdef HAVE_JASPER
        "JPEG-2000 files (*.jp2)\0*.jp2\0"
#endif
#ifdef HAVE_WEBP
        "WebP files (*.webp)\0*.webp\0"
#endif
        "Portable image format (*.pbm;*.pgm;*.ppm;*.pxm;*.pnm)\0*.pbm;*.pgm;*.ppm;*.pxm;*.pnm\0"
#ifdef HAVE_OPENEXR
        "OpenEXR Image files (*.exr)\0*.exr\0"
#endif
        "Radiance HDR (*.hdr;*.pic)\0*.hdr;*.pic\0"
        "Sun raster files (*.sr;*.ras)\0*.sr;*.ras\0"
        "All Files (*.*)\0*.*\0";
    ofn.lpstrFile = szFileName;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_EXPLORER | OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT | OFN_NOREADONLYRETURN | OFN_NOCHANGEDIR;
#ifdef HAVE_PNG
    ofn.lpstrDefExt = "png";
#else
    ofn.lpstrDefExt = "bmp";
#endif

    if (GetSaveFileName(&ofn))
    {
        alchemy::Matrix tmp;
//        cv::flip(cv::Mat(sz.cy, sz.cx, Z_8UC(channels), data, (sz.cx * channels + 3) & -4), tmp, 0);
        alchemy::imwrite(szFileName, tmp);
    }
}

int zWaitKey(int delay)
{
    int time0 = GetTickCount();

    for (;;)
    {
        MSG message;
        int is_processed = 0;

        if ((delay > 0 && abs((int)(GetTickCount() - time0)) >= delay) || hg_windows == 0)
            return -1;

        if (delay <= 0)
            GetMessage(&message, 0, 0, 0);
        else if (PeekMessage(&message, 0, 0, 0, PM_REMOVE) == FALSE)
        {
            Sleep(1);
            continue;
        }

        for (zWindow * window = hg_windows; window != 0 && is_processed == 0; window = window->next)
        {
            if (window->hwnd == message.hwnd || window->frame == message.hwnd)
            {
                is_processed = 1;
                switch (message.message)
                {
                case WM_DESTROY:
                case WM_CHAR:
                    DispatchMessage(&message);
                    return (int)message.wParam;

                case WM_SYSKEYDOWN:
                    if (message.wParam == VK_F10)
                    {
                        is_processed = 1;
                        return (int)(message.wParam << 16);
                    }
                    break;

                case WM_KEYDOWN:
                    TranslateMessage(&message);
                    if ((message.wParam >= VK_F1 && message.wParam <= VK_F24) ||
                        message.wParam == VK_HOME || message.wParam == VK_END ||
                        message.wParam == VK_UP || message.wParam == VK_DOWN ||
                        message.wParam == VK_LEFT || message.wParam == VK_RIGHT ||
                        message.wParam == VK_INSERT || message.wParam == VK_DELETE ||
                        message.wParam == VK_PRIOR || message.wParam == VK_NEXT)
                    {
                        DispatchMessage(&message);
                        is_processed = 1;
                        return (int)(message.wParam << 16);
                    }

                    // Intercept Ctrl+C for copy to clipboard
                    if ('C' == message.wParam && (::GetKeyState(VK_CONTROL) >> 15))
                        ::SendMessage(message.hwnd, WM_COPY, 0, 0);

                    // Intercept Ctrl+S for "save as" dialog
                    if ('S' == message.wParam && (::GetKeyState(VK_CONTROL) >> 15))
                        showSaveDialog(window);

                default:
                    DispatchMessage(&message);
                    is_processed = 1;
                    break;
                }
            }
        }

        if (!is_processed)
        {
            TranslateMessage(&message);
            DispatchMessage(&message);
        }
    }
}


static zTrackbar* zFindTrackbarByName(const zWindow* window, const char* name)
{
    auto trackbar = window->toolbar.first;

    for (; trackbar != 0 && strcmp(trackbar->name, name) != 0; trackbar = trackbar->next)
        ;

    return trackbar;
}


static int zCreateTrackbar(const char* trackbar_name, const char* window_name,
    int* val, int count, zTrackbarCallback on_notify,
    zTrackbarCallback2 on_notify2, void* userdata)
{
    int result = 0;
    char slider_name[32];
    zWindow* window = 0;
    zTrackbar* trackbar = 0;
    int pos = 0;

	assert(window_name && trackbar_name);
	assert(count >= 0);

    window = zFindWindowByName(window_name);
    if (!window)
        return 0;

    trackbar = zFindTrackbarByName(window, trackbar_name);
    if (!trackbar)
    {
        TBBUTTON tbs = {};
        TBBUTTONINFO tbis = {};
        RECT rect;
        int bcount;
        int len = (int)strlen(trackbar_name);

        // create toolbar if it is not created yet
        if (!window->toolbar.toolbar)
        {
            const int default_height = 30;

            // CreateToolbarEx is deprecated and forces linking against Comctl32.lib.
            window->toolbar.toolbar = CreateWindowEx(0, TOOLBARCLASSNAME, NULL,
                WS_CHILD | CCS_TOP | TBSTYLE_WRAPABLE | BTNS_AUTOSIZE | BTNS_BUTTON,
                0, 0, 0, 0,
                window->frame, NULL, GetModuleHandle(NULL), NULL);
            // CreateToolbarEx automatically sends this but CreateWindowEx doesn't.
            SendMessage(window->toolbar.toolbar, TB_BUTTONSTRUCTSIZE, (WPARAM)sizeof(TBBUTTON), 0);

            GetClientRect(window->frame, &rect);
            MoveWindow(window->toolbar.toolbar, 0, 0,
                rect.right - rect.left, default_height, TRUE);
            SendMessage(window->toolbar.toolbar, TB_AUTOSIZE, 0, 0);
            ShowWindow(window->toolbar.toolbar, SW_SHOW);

            window->toolbar.first = 0;
            window->toolbar.pos = 0;
            window->toolbar.rows = 0;
            window->toolbar.toolBarProc =
                (WNDPROC)GetWindowLong(window->toolbar.toolbar, GWL_WNDPROC);

            zUpdateWindowPos(window);

            // Subclassing from toolbar
            SetWindowLong(window->toolbar.toolbar, GWL_WNDPROC, (size_t)HGToolbarProc);
            SetWindowLong(window->toolbar.toolbar, GWL_USERDATA, (size_t)window);
        }

        /* Retrieve current buttons count */
        bcount = (int)SendMessage(window->toolbar.toolbar, TB_BUTTONCOUNT, 0, 0);

        if (bcount > 1)
        {
            /* If this is not the first button then we need to
            separate it from the previous one */
            tbs.iBitmap = 0;
            tbs.idCommand = bcount; // Set button id to it's number
            tbs.iString = 0;
            tbs.fsStyle = TBSTYLE_SEP;
            tbs.fsState = TBSTATE_ENABLED;
            SendMessage(window->toolbar.toolbar, TB_ADDBUTTONS, 1, (LPARAM)&tbs);

            // Retrieve current buttons count
            bcount = (int)SendMessage(window->toolbar.toolbar, TB_BUTTONCOUNT, 0, 0);
        }

        /* Add a button which we're going to cover with the slider */
        tbs.iBitmap = 0;
        tbs.idCommand = bcount; // Set button id to it's number
        tbs.fsState = TBSTATE_ENABLED;
#if 0/*!defined WIN64 && !defined EM64T*/
        tbs.fsStyle = 0;
        tbs.iString = 0;
#else

#ifndef TBSTYLE_AUTOSIZE
#define TBSTYLE_AUTOSIZE        0x0010
#endif

#ifndef TBSTYLE_GROUP
#define TBSTYLE_GROUP           0x0004
#endif
        //tbs.fsStyle = TBSTYLE_AUTOSIZE;
        tbs.fsStyle = TBSTYLE_GROUP;
        tbs.iString = (INT_PTR)trackbar_text;
#endif
        SendMessage(window->toolbar.toolbar, TB_ADDBUTTONS, 1, (LPARAM)&tbs);

        /* Adjust button size to the slider */
        tbis.cbSize = sizeof(tbis);
        tbis.dwMask = TBIF_SIZE;

        GetClientRect(window->hwnd, &rect);
        tbis.cx = (unsigned short)(rect.right - rect.left);

        SendMessage(window->toolbar.toolbar, TB_SETBUTTONINFO,
            (WPARAM)tbs.idCommand, (LPARAM)&tbis);

        /* Get button pos */
        SendMessage(window->toolbar.toolbar, TB_GETITEMRECT,
            (WPARAM)tbs.idCommand, (LPARAM)&rect);

        /* Create a slider */
        trackbar = (zTrackbar*)malloc(sizeof(zTrackbar) + len + 1);
        trackbar->signature = Z_TRACKBAR_MAGIC_VAL;
        trackbar->notify = 0;
        trackbar->notify2 = 0;
        trackbar->parent = window;
        trackbar->pos = 0;
        trackbar->data = 0;
        trackbar->id = bcount;
        trackbar->next = window->toolbar.first;
        trackbar->name = (char*)(trackbar + 1);
        memcpy(trackbar->name, trackbar_name, len + 1);
        window->toolbar.first = trackbar;

        sprintf(slider_name, "Trackbar%p", val);
        trackbar->hwnd = CreateWindowEx(0, TRACKBAR_CLASS, slider_name,
            WS_CHILD | WS_VISIBLE | TBS_AUTOTICKS |
            TBS_FIXEDLENGTH | TBS_HORZ | TBS_BOTTOM,
            rect.left + HG_BUDDY_WIDTH, rect.top,
            rect.right - rect.left - HG_BUDDY_WIDTH,
            rect.bottom - rect.top, window->toolbar.toolbar,
            (HMENU)(size_t)bcount, hg_hinstance, 0);

        sprintf(slider_name, "Buddy%p", val);
        trackbar->buddy = CreateWindowEx(0, "STATIC", slider_name,
            WS_CHILD | SS_RIGHT,
            rect.left, rect.top,
            HG_BUDDY_WIDTH, rect.bottom - rect.top,
            window->toolbar.toolbar, 0, hg_hinstance, 0);

        SetWindowLong(trackbar->hwnd, GWL_USERDATA, (size_t)trackbar);

        /* Minimize the number of rows */
        SendMessage(window->toolbar.toolbar, TB_SETROWS,
            MAKEWPARAM(1, FALSE), (LPARAM)&rect);
    }
    else
    {
        trackbar->data = 0;
        trackbar->notify = 0;
        trackbar->notify2 = 0;
    }

    trackbar->maxval = count;

    /* Adjust slider parameters */
    SendMessage(trackbar->hwnd, TBM_SETRANGEMIN, (WPARAM)TRUE, (LPARAM)0);
    SendMessage(trackbar->hwnd, TBM_SETRANGEMAX, (WPARAM)TRUE, (LPARAM)count);
    SendMessage(trackbar->hwnd, TBM_SETTICFREQ, (WPARAM)1, (LPARAM)0);
    if (val)
        pos = *val;

    SendMessage(trackbar->hwnd, TBM_SETPOS, (WPARAM)TRUE, (LPARAM)pos);
    SendMessage(window->toolbar.toolbar, TB_AUTOSIZE, 0, 0);

    trackbar->pos = -1;
    zUpdateTrackbar(trackbar, pos);
    ShowWindow(trackbar->buddy, SW_SHOW);
    ShowWindow(trackbar->hwnd, SW_SHOW);

    trackbar->notify = on_notify;
    trackbar->notify2 = on_notify2;
    trackbar->userdata = userdata;
    trackbar->data = val;

    /* Resize the window to reflect the toolbar resizing*/
    zUpdateWindowPos(window);

    result = 1;

    return result;
}

int zCreateTrackbar(const char* trackbar_name, const char* window_name,
    int* val, int count, zTrackbarCallback on_notify)
{
    return zCreateTrackbar(trackbar_name, window_name, val, count,
        on_notify, 0, 0);
}

int zCreateTrackbar2(const char* trackbar_name, const char* window_name,
    int* val, int count, zTrackbarCallback2 on_notify2,
    void* userdata)
{
    return zCreateTrackbar(trackbar_name, window_name, val, count,
        0, on_notify2, userdata);
}

void zSetMouseCallback(const char* window_name, zMouseCallback on_mouse, void* param)
{
    zWindow* window = nullptr;

    if (!window_name)
        _log_( "NULL window name");

    window = zFindWindowByName(window_name);
    if (!window)
        return;

    window->on_mouse = on_mouse;
    window->on_mouse_param = param;
}


int zGetTrackbarPos(const char* trackbar_name, const char* window_name)
{
    int pos = -1;

    zTrackbar* trackbar = nullptr;

    if (trackbar_name == 0 || window_name == 0)
        _log_("NULL trackbar or window name");

    auto window = zFindWindowByName(window_name);
    if (window)
        trackbar = zFindTrackbarByName(window, trackbar_name);

    if (trackbar)
        pos = trackbar->pos;

    return pos;
}


void zSetTrackbarPos(const char* trackbar_name, const char* window_name, int pos)
{
    zTrackbar* trackbar = nullptr;

    if (trackbar_name == 0 || window_name == 0)
        _log_("NULL trackbar or window name");

    auto window = zFindWindowByName(window_name);
    if (window)
        trackbar = zFindTrackbarByName(window, trackbar_name);

    if (trackbar) {
        if (pos < 0)
            pos = 0;

        if (pos > trackbar->maxval)
            pos = trackbar->maxval;

        SendMessage(trackbar->hwnd, TBM_SETPOS, (WPARAM)TRUE, (LPARAM)pos);
        zUpdateTrackbar(trackbar, pos);
    }
}


void zSetTrackbarMax(const char* trackbar_name, const char* window_name, int maxval)
{
    if (maxval >= 0) {
        zWindow* window = nullptr;
        zTrackbar* trackbar = nullptr;
        if (trackbar_name == 0 || window_name == 0) {
            _log_("NULL trackbar or window name");
        }

        window = zFindWindowByName(window_name);
        if (window) {
            trackbar = zFindTrackbarByName(window, trackbar_name);
            if (trackbar)
            {
                // The position will be min(pos, maxval).
                trackbar->maxval = (trackbar->minval>maxval) ? trackbar->minval : maxval;
                SendMessage(trackbar->hwnd, TBM_SETRANGEMAX, (WPARAM)TRUE, (LPARAM)maxval);
            }
        }
    }
}


void zSetTrackbarMin(const char* trackbar_name, const char* window_name, int minval)
{
    if (minval >= 0) {
        zWindow* window = nullptr;
        zTrackbar* trackbar = nullptr;

        if (trackbar_name == 0 || window_name == 0) {
            _log_("NULL trackbar or window name");
        }

        window = zFindWindowByName(window_name);
        if (window) {
            trackbar = zFindTrackbarByName(window, trackbar_name);
            if (trackbar) {
                // The position will be min(pos, maxval).
                trackbar->minval = (minval<trackbar->maxval) ? minval : trackbar->maxval;
                SendMessage(trackbar->hwnd, TBM_SETRANGEMIN, (WPARAM)TRUE, (LPARAM)minval);
            }
        }
    }
}


void* zGetWindowHandle(const char* window_name)
{
    void* hwnd = nullptr;

    if (window_name == 0)
        _log_("NULL window name");

    auto window = zFindWindowByName(window_name);
    if (window)
        hwnd = (void*)window->hwnd;

    return hwnd;
}


const char* cvGetWindowName(void* window_handle)
{
	assert(window_handle != 0);

	auto window_name = "";

	const auto window = zWindowByHWND(static_cast<HWND>(window_handle));
    if (window)
        window_name = window->name;

    return window_name;
}

#endif//!_WIN32
