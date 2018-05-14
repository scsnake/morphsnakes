import Queue
import Tkinter as tk
# ------------------------------------------------------------------------------
# Code to simulate background process periodically updating the image file.
# Note:
#   It's important that this code *not* interact directly with tkinter
#   stuff in the main process since it doesn't support multi-threading.
import threading

import mouse
import mss
import numpy as np
from PIL import Image, ImageTk
from skimage.morphology import binary_dilation
from skimage import measure
from time import time
import morphsnakes


def all_monitors_info():
    # return sorted(get_monitors(), key=lambda m: (m.x, m.y))
    with mss.mss() as sct:
        return sorted(sct.monitors[1:], key=lambda m: (m['left'], m['top']))


def mouse_in_which_monitor(x=None, y=None, monitors_info=None):
    if x is None or y is None:
        # x, y = pyautogui.position()
        x, y = mouse.get_position()
    if monitors_info is None:
        monitors_info = all_monitors_info()

    for i, m in enumerate(monitors_info):
        if m['left'] <= x < m['left'] + m['width'] and m['top'] <= y < m['top'] + m['height']:
            return i, m

    return -1, None


def get_current_screen(monitor=None):
    if monitor is None:
        which, monitor = mouse_in_which_monitor()
    with mss.mss() as sct:
        return sct.grab(monitor)


def circle_levelset(shape, center, sqradius, scalerow=1.0):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum((grid.T) ** 2, 0))
    u = np.float_(phi > 0)
    return u


def rgb2gray(img):
    """Convert a RGB image to gray scale."""
    return 0.2989 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]


class RegionGrowing:
    def __init__(self):
        self.mouseX, self.mouseY = mouse.get_position()
        self.which_monitor, self.this_monitor = mouse_in_which_monitor(self.mouseX, self.mouseY)
        self.data = rgb2gray(np.array(get_current_screen(self.this_monitor)))
        self.x, self.y = self.mouseX - self.this_monitor['left'], self.mouseY - self.this_monitor['top']
        # print 'mouse x,y = %d,%d' % (self.mouseX, self.mouseY)
        print 'x,y = %d,%d' % (self.x, self.y)
        self.region = np.ma.masked_array(self.data, np.ones(self.data.shape))
        self.last_added = []

        self.neighbor_matrix = np.ones((3, 3))
        self.neighbor_matrix[1, 1] = 0
        self.neighbors = [(c[0] - 1, c[1] - 1) for c, e in np.ndenumerate(self.neighbor_matrix) if e]
        # print self.x, self.y
        self.region.mask[self.y, self.x] = False
        for x, y in self.get_neighbors(self.x, self.y):
            try:
                self.region.mask[y, x] = False
            except:
                pass
            self.last_added.append((x, y))
        self.region_hx = Queue.LifoQueue()
        self.region_hx.put(np.copy(self.region.mask), 0)
        self.last_added_hx = Queue.LifoQueue()
        self.last_added_hx.put(np.copy(self.last_added), 0)
        self.growing_interval = 0.4

        self.growing_timer = None

        self.tmp = 0

    def get_neighbors(self, x, y):
        for i, j in self.neighbors:
            yield (x + i, y + j)

    def growing(self):
        if not mouse.is_pressed():
            self.growing_state = False
            return
        self.growing_state = True
        print 'growing'
        # self.tmp+=1
        # if self.tmp==2:
        #     tt=1
        # elif self.tmp>2:
        #     return
        mean = np.ma.mean(self.region)
        std = np.ma.std(self.region)
        if std < 0.001:
            std = 20
        added_points = []
        for last_x, last_y in self.last_added:
            for x, y in self.get_neighbors(last_x, last_y):
                try:
                    if self.region.mask[y, x] and (self.data[y, x] - mean) < std * 0.5:
                        self.region.mask[y, x] = False
                        added_points.append((x, y))
                        # print 'add %d' % (int(self.data[y,x]), )
                        # elif self.region.mask[y, x]:
                        #     print (self.data[y, x] - mean) < std * 0.5
                except:
                    pass
        # print mean, std
        self.region_hx.put(np.copy(self.region.mask), 0)
        self.last_added_hx.put(added_points, 0)
        self.last_added = added_points
        # print added_points

        # threading.Thread(target=self.overlay, args=(((~self.region.mask) * 255).astype(np.uint8),)).start()
        # print np.count_nonzero(~self.region.mask)
        self.overlay(((~self.region.mask) * 255).astype(np.uint8))

        if mouse.is_pressed():
            self.growing_timer = threading.Timer(self.growing_interval, self.growing)
            self.growing_timer.start()

    def finish(self):
        try:
            self.growing_timer.cancel()
        except:
            pass
        # print 'Finish'
        self.growing_state = False
        self.overlay(((~self.region.mask) * 255).astype(np.uint8))
        # print np.nonzero(~self.region.mask)

        # threading.Thread(target=self.overlay, args=(((~self.region.mask) * 255).astype(np.uint8),)).start()

    def prior_step(self):
        try:
            self.region.mask = self.region_hx.get(0)
            self.last_added = self.last_added_hx.get(0)
            self.overlay(((~self.region.mask) * 255).astype(np.uint8))
        except:
            print 'No prior step available!'

    def overlay(self, img):
        global q

        # im=ImageTk.PhotoImage(Image.open(r"C:\Users\Administrator\Downloads\example.png"))
        im = ImageTk.PhotoImage(Image.fromarray(img, 'L'))
        q.put(im, 0)
        q.put('show', 0)
        # overlay.show_image(im)
        # Image.fromarray(img, 'L').show()
        # threading.Thread(target=overlay.show_image, args=(im,)).start()
        # overlay.show_image(im)

    def hide_after(self, ms=1000):
        global q
        try:
            self.hide_after_thread.cancel()
        except:
            pass
        th = threading.Timer(ms / 1000.0, q.put, ['hide', 0])
        self.hide_after_thread = th
        th.start()

class Morph:
    def __init__(self, img, seed= (0,0), alpha=1000, sigma=5.48, smoothing=1, threshold=0.31, balloon=20):
        self.alpha = alpha
        self.sigma = sigma
        self.smoothing = smoothing
        self.threshold = threshold
        self.balloon = balloon
        self.img = img
        self.seed = seed
        self.gI = morphsnakes.gborders(img, alpha=alpha, sigma=sigma)
        self.mgac = morphsnakes.MorphGAC(self.gI, smoothing=smoothing, threshold=threshold, balloon=balloon)
        self.last_crop_levelset= self.last_levelset = self.mgac.levelset = circle_levelset(img.shape, (seed[1], seed[0]), balloon)
        self.last_crop_bounds= (0,0,1,1)

    def step(self):
        balloon = self.balloon
        img = self.img
        # Coordinates of non-black pixels.
        coords = np.argwhere(self.last_levelset)

        # Bounding box of non-black pixels.
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0) + 1  # slices are exclusive at the top
        bounds = [x0, y0, x1, y1]
        crop_bounds = []
        for i, c in enumerate(bounds):
            c+=balloon*2 if i>1 else balloon*(-2)
            if i<2:
                c=0 if c<0 else c
            elif i==2:
                c=img.shape[0] if c>img.shape[0] else c
            else:
                c = img.shape[1] if c > img.shape[1] else c
            crop_bounds.append(c)
        print self.last_crop_levelset.shape
        gI = morphsnakes.gborders(img[x0:x1, y0:y1], alpha=self.alpha, sigma=self.sigma)
        mgac = morphsnakes.MorphGAC(gI, smoothing=self.smoothing, threshold=self.threshold, balloon=self.balloon)
        mgac.levelset = self.last_levelset[x0:x1, y0:y1]

        mgac.step()
        self.last_crop_levelset = np.copy(self.last_levelset[x0:x1, y0:y1])
        self.last_levelset[x0:x1, y0:y1] = mgac.levelset
        self.last_crop_bounds = tuple(crop_bounds)





class RegionGrowingMorph:
    def __init__(self):
        global q
        self.mouseX, self.mouseY = mouse.get_position()
        self.which_monitor, self.this_monitor = mouse_in_which_monitor(self.mouseX, self.mouseY)
        self.data = rgb2gray(np.array(get_current_screen(self.this_monitor)))

        # print self.mouseX, self.mouseY, self.which_monitor, self.this_monitor, self.data.shape

        self.x, self.y = self.mouseX - self.this_monitor['left'], self.mouseY - self.this_monitor['top']
        # print 'mouse x,y = %d,%d' % (self.mouseX, self.mouseY)
        print 'x,y = %d,%d' % (self.x, self.y)
        self.region = np.ma.masked_array(self.data, np.ones(self.data.shape))
        self.last_added = []

        self.neighbor_matrix = np.ones((3, 3))
        self.neighbor_matrix[1, 1] = 0
        self.neighbors = [(c[0] - 1, c[1] - 1) for c, e in np.ndenumerate(self.neighbor_matrix) if e]
        # print self.x, self.y
        self.region.mask[self.y, self.x] = False
        for x, y in self.get_neighbors(self.x, self.y):
            try:
                self.region.mask[y, x] = False
            except:
                pass
            self.last_added.append((x, y))
        self.region_hx = Queue.LifoQueue()
        self.region_hx.put(np.copy(self.region.mask), 0)
        self.last_added_hx = Queue.LifoQueue()
        self.last_added_hx.put(np.copy(self.last_added), 0)
        self.growing_interval = 0.05

        self.growing_timer = None

        self.tmp = 0
        self.info_text = ''
        self.init_time = time()

        self.morph = Morph(self.data, (self.y, self.x))
        self.last_levelset=self.morph.last_levelset



    def get_neighbors(self, x, y):
        for i, j in self.neighbors:
            yield (x + i, y + j)

    def growing(self):
        # print mouse.is_pressed()
        if not mouse.is_pressed():
            self.growing_state = False
            return self.finish()
        self.growing_state = True
        # print 'growing'

        time_elapsed = time()-self.init_time
        iters=2**int(time_elapsed/2)
        if iters>10:
            iters=10

        for _ in range(iters):
            self.morph.step()



        if np.array_equal(self.last_levelset, self.morph.last_crop_levelset):
            print 'growing done!'

            return self.finish()
        else:
            ls = np.copy(self.morph.last_crop_levelset)
            self.last_levelset = ls
            self.region_hx.put(ls, 0)

        # threading.Thread(target=self.overlay, args=(((~self.region.mask) * 255).astype(np.uint8),)).start()
        # print np.count_nonzero(~self.region.mask)
        self.overlay((ls * 255).astype(np.uint8))

        if mouse.is_pressed():
            self.growing_timer = threading.Timer(self.growing_interval, self.growing)
            self.growing_timer.start()
        else:
            self.finish()

    def finish(self):
        try:
            self.growing_timer.cancel()
        except:
            pass
        print 'Finish'
        self.growing_state = False

        result = binary_dilation(self.morph.last_crop_levelset)
        regions = measure.regionprops(measure.label(result))

        result = (result * 255).astype(np.uint8)

        try:
            self.info_text = 'Diameter: %d * %d \nArea: %d' % (int(regions[0].minor_axis_length),
                                                               int(regions[0].major_axis_length),
                                                               regions[0].area)
        except:
            self.info_text=''

        self.overlay(result)
        self.hide_after()
        # print np.nonzero(~self.region.mask)

        # threading.Thread(target=self.overlay, args=(((~self.region.mask) * 255).astype(np.uint8),)).start()

    def prior_step(self):
        try:
            self.region.mask = self.region_hx.get(0)
            self.overlay()
        except:
            print 'No prior step available!'

    def overlay(self, img=None):
        global q

        # im=ImageTk.PhotoImage(Image.open(r"C:\Users\Administrator\Downloads\example.png"))

        if img is None:
            img = (self.morph.last_crop_levelset * 255).astype(np.uint8)


        if self.which_monitor>0:
            im = ImageTk.PhotoImage(Image.fromarray(img))
        else:
            rgb = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
            # rgb[:, :, 0] = img
            rgb[:, :, 1] = img
            im = ImageTk.PhotoImage(Image.fromarray(rgb))
        q.put(im, 0)
        q.put('show', 0)
        # overlay.show_image(im)
        # Image.fromarray(img, 'L').show()
        # threading.Thread(target=overlay.show_image, args=(im,)).start()
        # overlay.show_image(im)

    def hide_after(self, ms=10000):
        global q
        try:
            self.hide_after_thread.cancel()
        except:
            pass
        th = threading.Timer(ms / 1000.0, q.put, ['hide', 0])
        self.hide_after_thread = th
        th.start()


def rbutton_down(*args):
    global rg
    if not rg or rg.growing_state:
        return
    rg.prior_step()
    rg.hide_after()


def lbutton_down(*args):
    global q, rg, task
    # canvas.config(width=1000, height=700)
    # if not keyboard.is_pressed('shift'):
    #     return
    try:
        task.cancel()
    except:
        pass

    task = threading.Timer(0.5, RegionGrowingTask)
    task.start()
    # q.put(ImageTk.PhotoImage(Image.fromarray((np.eye(800) * 255).astype('uint8'), 'L').resize((800,800))),0)


def RegionGrowingTask():
    global rg
    if mouse.is_pressed():
        rg = RegionGrowingMorph()
        rg.growing()


def lbutton_up(*args):
    global q, rg
    if not rg:
        return
    rg.finish()
    rg.hide_after()
    # threading.Threaad(target=lambda rg: rg.finish(), args=(rg,)).start()





class Overlay:
    def __init__(self):
        root = tk.Tk()

        frame = tk.Frame(root, bg='black')
        canvas = tk.Canvas(frame, bg="black", bd=0, highlightthickness=0)
        img = None  # initially only need a canvas image place-holder
        # img = ImageTk.PhotoImage(Image.open(r"C:\Users\Administrator\Downloads\example.png"))
        image_id = canvas.create_image(0, 0, image=img, anchor='nw')
        canvas.pack(side="left", fill="both", expand=True)

        info = tk.Label(root, text="", width=1, borderwidth=1, font=('Verdana', 20), justify='left',
                        bg='white', fg='gray')
        info.place(relx=0.0, rely=0.0, x=-2, y=-2, anchor="se")

        frame.pack(side="top", fill="both", expand=True)

        root.title('overlay')
        root.overrideredirect(1)
        root.wm_attributes("-topmost", True)
        root.wm_attributes("-alpha", 0.5)
        root.wm_attributes("-toolwindow", True)
        root.wm_attributes("-disabled", True)
        root.wm_attributes("-transparentcolor", "black")
        # root.bind('<ButtonRelease-1>', lbutton_up)

        self.root = root
        self.frame = frame
        self.info = info
        self.canvas = canvas
        self.image_id = image_id
        self.x = self.y = 0
        self.refresh_image()

    def refresh_image(self):
        global rg, q
        root = self.root
        canvas = self.canvas
        info = self.info
        frame = self.frame
        image_id = self.image_id
        try:
            img = q.get(0)
            # print img
            if img == 'show':
                root.update()
                root.deiconify()
            elif img == 'hide' and not rg.growing_state:
                info.config(text=rg.info_text, width=1)
                info.place(relx=0.0, rely=0.0, x=-2, y=-2, anchor="nw")
                root.withdraw()
            else:
                bounds = rg.morph.last_crop_bounds
                w, h = img.width(), img.height()
                x, y = rg.this_monitor['left']+bounds[0], rg.this_monitor['top']+bounds[1]
                # print x, y

                x=str(x) if x<0 else '+'+str(x)
                y=str(y) if y<0 else '+'+str(y)
                root.geometry('%dx%d%s%s' % (w+300, h+300, x, y))
                canvas.image = img

                canvas.config(width=w, height=h)
                # print img.width(), img.height()
                canvas.itemconfigure(image_id, image=img)

                if rg.info_text:
                    print rg.info_text
                    info.config(text=rg.info_text, width=20)
                    info.place(relx=1.0 , rely=1.0)
        except:  # missing or corrupt image file
            pass
        # repeat every half sec
        canvas.after(10, self.refresh_image)


if __name__ == '__main__':
    rg = None
    task = None
    # root = tk.Tk()
    q = Queue.Queue()
    # ------------------------------------------------------------------------------
    # More code to simulate background process periodically updating the image file.
    # th = threading.Thread(target=update_image_file, args=(q,))
    # th.daemon = True  # terminates whenever main thread does
    # th.start()
    # while not os.path.exists(image_path):  # let it run until image file exists
    #     time.sleep(.1)
    # ------------------------------------------------------------------------------


    mouse.on_button(rbutton_down, buttons='right', types='down')
    mouse.on_button(lbutton_down, buttons='left', types='down')
    # mouse.on_button(lbutton_up, buttons='left', types='up')

    overlay = Overlay()
    overlay.root.mainloop()
