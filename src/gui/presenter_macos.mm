/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/

#define _DARWIN_C_SOURCE

/*
 * the Quartz presenter uses two non-interactive Core Animation layers above
 * GTK's view. the image layer carries linear Rec.2020 float pixels, so
 * WindowServer performs the display profile conversion and EDR mapping. the
 * foreground layer carries the 8-bit cairo replay of everything drawn above
 * the image. see dev-doc/display_presentation.md.
 */

#include "gui/presenter_layers.h"
#include "common/darktable.h"
#include "common/math.h"

#include <AppKit/AppKit.h>
#include <CoreGraphics/CoreGraphics.h>
#include <QuartzCore/QuartzCore.h>
#include <gdk/gdkquartz.h>

#if __has_include(<gdk/quartz/gdkquartz-cocoa-access.h>)
extern "C" {
#include <gdk/quartz/gdkquartz-cocoa-access.h>
}
#endif

#include <cmath>
#include <cstring>

// EDR is expressed relative to the reference white in the rendered pipe.
// Quantizing to 1/8 EV prevents display transitions from reprocessing the
// whole darkroom for every intermediate value reported by macOS.
static float _quantized_headroom(NSScreen *screen)
{
  CGFloat value = screen ? screen.maximumPotentialExtendedDynamicRangeColorComponentValue : 1.0;
  if(!std::isfinite(value) || value < 1.0) value = 1.0;
  return (float)exp2(round(log2((double)value) * 8.0) / 8.0);
}

typedef struct _widget_t
{
  CALayer *image;
  CALayer *foreground;
  gboolean has_image;
} _widget_t;

typedef struct _backend_t
{
  dt_presenter_backend_t backend;
  dt_presenter_window_t *window;
  GtkWidget *toplevel;
  NSView *view;
  NSWindow *native_window;
  CALayer *host;
  GdkFrameClock *clock;
  gulong paint_handler;
  id screen_observer;
  id display_observer;
  id window_observer;
  dt_display_output_t output;
} _backend_t;

static void _provider_release(void *info,
                              const void *data,
                              size_t size)
{
  (void)info;
  (void)size;
  g_free((void *)data);
}

static uint16_t _float_to_half(const float value)
{
  if(value < 6.103515625e-05f) return (uint16_t)lrintf(value * 16777216.0f);
  union { float f; uint32_t u; } bits = { .f = value };
  const uint32_t rounded = bits.u + 0x0fff + ((bits.u >> 13) & 1);
  return (uint16_t)((rounded >> 13) - ((127 - 15) << 10));
}

static CGImageRef _image_create(const dt_presenter_layers_t *layers,
                                const float headroom,
                                float *content_headroom)
{
  const size_t npixels = (size_t)layers->width * layers->height;
  const size_t size = npixels * 4 * sizeof(uint16_t);
  uint16_t *data = (uint16_t *)g_try_malloc(size);
  if(!data) return NULL;

  const float *const input = layers->image;
  float peak = 1.0f;
  DT_OMP_FOR(reduction(max : peak))
  for(size_t k = 0; k < npixels; k++)
  {
    const float alpha = input[4 * k + 3];
    if(!(alpha > 0.0f))
    {
      memset(data + 4 * k, 0, 4 * sizeof(uint16_t));
      continue;
    }
    for(int c = 0; c < 3; c++)
    {
      const float v = input[4 * k + c];
      const float value = dt_isnan(v) ? 0.0f : CLAMPF(v, 0.0f, headroom);
      data[4 * k + c] = _float_to_half(value);
      peak = fmaxf(peak, value);
    }
    data[4 * k + 3] = _float_to_half(1.0f);
  }
  *content_headroom = peak;

  CGDataProviderRef provider =
    CGDataProviderCreateWithData(NULL, data, size, _provider_release);
  if(!provider)
  {
    g_free(data);
    return NULL;
  }

  CGColorSpaceRef space = CGColorSpaceCreateWithName(kCGColorSpaceExtendedLinearITUR_2020);
  const CGBitmapInfo bitmap_info =
    (CGBitmapInfo)(kCGImageAlphaLast | kCGBitmapFloatComponents | kCGBitmapByteOrder16Host);
  CGImageRef image = NULL;
  if(space)
  {
    if(@available(macOS 15.0, *))
    {
      image = CGImageCreateWithContentHeadroom(*content_headroom,
                                               layers->width,
                                               layers->height,
                                               16,
                                               64,
                                               (size_t)layers->width * 8,
                                               space,
                                               bitmap_info,
                                               provider,
                                               NULL,
                                               false,
                                               kCGRenderingIntentRelativeColorimetric);
    }
    if(!image)
    {
      image = CGImageCreate(layers->width,
                            layers->height,
                            16,
                            64,
                            (size_t)layers->width * 8,
                            space,
                            bitmap_info,
                            provider,
                            NULL,
                            false,
                            kCGRenderingIntentRelativeColorimetric);
    }
    CGColorSpaceRelease(space);
  }
  CGDataProviderRelease(provider);
  return image;
}

static CGImageRef _foreground_create(const dt_presenter_layers_t *layers)
{
  const size_t size = (size_t)layers->width * layers->height * 4;
  uint8_t *data = (uint8_t *)g_try_malloc(size);
  if(!data) return NULL;
  memcpy(data, layers->foreground, size);

  CGDataProviderRef provider =
    CGDataProviderCreateWithData(NULL, data, size, _provider_release);
  if(!provider)
  {
    g_free(data);
    return NULL;
  }
  CGColorSpaceRef space = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
  const CGBitmapInfo bitmap_info =
    (CGBitmapInfo)(kCGImageAlphaPremultipliedFirst | kCGBitmapByteOrder32Little);
  CGImageRef image = space
    ? CGImageCreate(layers->width,
                    layers->height,
                    8,
                    32,
                    (size_t)layers->width * 4,
                    space,
                    bitmap_info,
                    provider,
                    NULL,
                    false,
                    kCGRenderingIntentDefault)
    : NULL;
  if(space) CGColorSpaceRelease(space);
  CGDataProviderRelease(provider);
  return image;
}

static void _update_host(_backend_t *b)
{
  if(!b->host || !b->view) return;
  const NSRect bounds = [b->view bounds];
  [CATransaction begin];
  [CATransaction setDisableActions:YES];
  b->host.frame = CGRectMake(0, 0, bounds.size.width, bounds.size.height);
  [CATransaction commit];
}

static void _set_frame(_backend_t *b,
                       CALayer *layer,
                       const dt_presenter_layers_t *layers)
{
  const NSRect bounds = [b->view bounds];
  const CGFloat width = (CGFloat)layers->width / layers->scale;
  const CGFloat height = (CGFloat)layers->height / layers->scale;
  const CGFloat y = [b->host isGeometryFlipped]
    ? layers->y
    : bounds.size.height - layers->y - height;
  layer.frame = CGRectMake(layers->x, y, width, height);
  layer.contentsScale = layers->scale;
}

static void _set_dynamic_range(CALayer *layer,
                               const gboolean extended)
{
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
  if(@available(macOS 26.0, *))
  {
    layer.preferredDynamicRange = extended ? CADynamicRangeHigh : CADynamicRangeStandard;
    return;
  }
#endif
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  if(@available(macOS 10.11, *)) layer.wantsExtendedDynamicRangeContent = extended;
#pragma clang diagnostic pop
}

static void _set_output(_backend_t *b)
{
  NSScreen *screen = b->native_window.screen ?: NSScreen.mainScreen;
  dt_display_output_t output = {};
  output.type = DT_COLORSPACE_LIN_REC2020;
  output.intent = DT_INTENT_RELATIVE_COLORIMETRIC;
  output.headroom = _quantized_headroom(screen);
  output.managed = TRUE;
  dt_display_output_t gtk = {};
  gtk.type = DT_COLORSPACE_SRGB;
  gtk.intent = output.intent;
  gtk.headroom = 1.0f;
  gtk.managed = TRUE;
  if(dt_display_output_equal(&b->output, &output))
    return;
  b->output = output;
  dt_presenter_window_set_output(b->window, &output, &gtk);
}

// GTK3 draws widgets in the frame clock's paint phase and commits its
// Quartz surface after that phase. Present the Core Animation layers after
// GTK has finished drawing, so they become visible with the same frame.
static void _paint(GdkFrameClock *clock,
                   _backend_t *b)
{
  dt_presenter_window_present(b->window);
  (void)clock;
}

static void _screen_changed(NSNotification *notification,
                            _backend_t *b)
{
  _set_output(b);
  (void)notification;
}

static gboolean _present(dt_presenter_backend_t *self,
                         GtkWidget *widget,
                         void **widget_data,
                         const dt_presenter_layers_t *layers)
{
  _backend_t *b = (_backend_t *)self;
  _widget_t *w = (_widget_t *)*widget_data;
  if(!w)
  {
    w = (_widget_t *)g_malloc0(sizeof(_widget_t));
    if(!w) return FALSE;
    w->image = [CALayer layer];
    w->foreground = [CALayer layer];
    if(!w->image || !w->foreground)
    {
      g_free(w);
      return FALSE;
    }
    w->image.contentsGravity = kCAGravityResize;
    w->foreground.contentsGravity = kCAGravityResize;
    w->image.contentsFormat = kCAContentsFormatRGBA16Float;
    w->foreground.contentsFormat = kCAContentsFormatRGBA8Uint;
    w->image.opaque = NO;
    w->foreground.opaque = NO;
    [b->host addSublayer:w->image];
    [b->host addSublayer:w->foreground];
    *widget_data = w;
  }

  if(!layers->image && !w->has_image) return FALSE;

  CGImageRef image = NULL;
  float content_headroom = 1.0f;
  if(layers->image)
  {
    image = _image_create(layers, b->output.headroom, &content_headroom);
    if(!image) return FALSE;
  }
  CGImageRef foreground = _foreground_create(layers);
  if(!foreground)
  {
    if(image) CGImageRelease(image);
    return FALSE;
  }

  [CATransaction begin];
  [CATransaction setDisableActions:YES];
  _update_host(b);
  _set_frame(b, w->foreground, layers);
  w->foreground.contents = (id)foreground;
  w->foreground.hidden = NO;
  if(image)
  {
    _set_frame(b, w->image, layers);
    w->image.contents = (id)image;
    w->image.hidden = NO;
    _set_dynamic_range(w->image, content_headroom > 1.0f);
    w->has_image = TRUE;
  }
  [CATransaction commit];
  CGImageRelease(foreground);
  if(image) CGImageRelease(image);
  (void)widget;
  return TRUE;
}

static void _clear(dt_presenter_backend_t *self,
                   void **widget_data)
{
  _backend_t *b = (_backend_t *)self;
  _widget_t *w = (_widget_t *)*widget_data;
  if(!w) return;
  [CATransaction begin];
  [CATransaction setDisableActions:YES];
  w->image.hidden = YES;
  w->foreground.hidden = YES;
  w->image.contents = nil;
  w->foreground.contents = nil;
  [CATransaction commit];
  w->has_image = FALSE;
  (void)b;
}

static void _release(dt_presenter_backend_t *self,
                     void **widget_data)
{
  _backend_t *b = (_backend_t *)self;
  _widget_t *w = (_widget_t *)*widget_data;
  if(!w) return;
  [w->image removeFromSuperlayer];
  [w->foreground removeFromSuperlayer];
  g_free(w);
  *widget_data = NULL;
  (void)b;
}

static void _free(dt_presenter_backend_t *self)
{
  _backend_t *b = (_backend_t *)self;
  if(b->screen_observer)
    [[NSNotificationCenter defaultCenter] removeObserver:b->screen_observer];
  if(b->display_observer)
    [[NSNotificationCenter defaultCenter] removeObserver:b->display_observer];
  if(b->window_observer)
    [[NSNotificationCenter defaultCenter] removeObserver:b->window_observer];
  if(b->clock && b->paint_handler)
    g_signal_handler_disconnect(b->clock, b->paint_handler);
  [b->host removeFromSuperlayer];
  g_free(b);
}

dt_presenter_backend_t *dt_presenter_macos_new(GtkWidget *toplevel,
                                               dt_presenter_window_t *window)
{
  GdkWindow *gdk_window = gtk_widget_get_window(toplevel);
  if(!gdk_window || !GDK_IS_QUARTZ_WINDOW(gdk_window)) return NULL;
  NSView *view = gdk_quartz_window_get_nsview(gdk_window);
  NSWindow *native_window = gdk_quartz_window_get_nswindow(gdk_window);
  if(!view || !native_window) return NULL;
  [view setWantsLayer:YES];
  if(!view.layer) return NULL;

  _backend_t *b = (_backend_t *)g_malloc0(sizeof(_backend_t));
  if(!b) return NULL;
  b->window = window;
  b->toplevel = toplevel;
  b->view = view;
  b->native_window = native_window;
  b->host = [CALayer layer];
  if(!b->host)
  {
    g_free(b);
    return NULL;
  }
  b->host.geometryFlipped = YES;
  b->host.masksToBounds = NO;
  b->host.zPosition = 1.0;
  [view.layer addSublayer:b->host];
  _update_host(b);

  b->backend.present = _present;
  b->backend.clear = _clear;
  b->backend.release = _release;
  b->backend.free = _free;
  b->clock = gdk_window_get_frame_clock(gdk_window);
  if(!b->clock)
  {
    [b->host removeFromSuperlayer];
    g_free(b);
    return NULL;
  }
  b->paint_handler = g_signal_connect_after(b->clock, "paint", G_CALLBACK(_paint), b);
  memset(&b->output, 0, sizeof(b->output));

  b->screen_observer =
    [[NSNotificationCenter defaultCenter]
      addObserverForName:NSScreenColorSpaceDidChangeNotification
                  object:nil
                   queue:[NSOperationQueue mainQueue]
              usingBlock:^(NSNotification *note) { _screen_changed(note, b); }];
  b->display_observer = [[NSNotificationCenter defaultCenter]
    addObserverForName:NSApplicationDidChangeScreenParametersNotification
                object:nil
                 queue:[NSOperationQueue mainQueue]
            usingBlock:^(NSNotification *note) { _screen_changed(note, b); }];
  b->window_observer = [[NSNotificationCenter defaultCenter]
    addObserverForName:NSWindowDidChangeScreenNotification
                object:b->native_window
                queue:[NSOperationQueue mainQueue]
            usingBlock:^(NSNotification *note) { _screen_changed(note, b); }];

  _set_output(b);
  dt_print(DT_DEBUG_CONTROL, "[presenter macos] Core Animation EDR backend enabled");
  return &b->backend;
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
