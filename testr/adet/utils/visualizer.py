import numpy as np
import pickle
from detectron2.utils.visualizer import Visualizer
import matplotlib.colors as mplc
import matplotlib.font_manager as mfm

# 导入中文字符集
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))
from terediff.dataset.chinese_vocab import CTLABELS, VOCAB_SIZE

class TextVisualizer(Visualizer):
    def __init__(self, image, metadata, instance_mode, cfg):
        Visualizer.__init__(self, image, metadata, instance_mode=instance_mode)
        self.voc_size = VOCAB_SIZE + 1  # +1 for NULL_CHAR
        self.use_customer_dictionary = cfg.MODEL.BATEXT.CUSTOM_DICT
        self.use_polygon = cfg.MODEL.TRANSFORMER.USE_POLYGON
        
        if not self.use_customer_dictionary:
            self.CTLABELS = CTLABELS
        else:
            with open(self.use_customer_dictionary, 'rb') as fp:
                self.CTLABELS = pickle.load(fp)
                
        assert(int(self.voc_size - 1) == len(self.CTLABELS)), \
            f"voc_size is not matched dictionary size, got {int(self.voc_size - 1)} and {len(self.CTLABELS)}."

    def draw_instance_predictions(self, predictions):
        if self.use_polygon:
            ctrl_pnts = predictions.polygons.numpy()
        else:
            ctrl_pnts = predictions.beziers.numpy()
        scores = predictions.scores.tolist()
        recs = predictions.recs

        self.overlay_instances(ctrl_pnts, recs, scores)

        return self.output

    def _ctrl_pnt_to_poly(self, pnt):
        if self.use_polygon:
            points = pnt.reshape(-1, 2)
        else:
            # bezier to polygon
            u = np.linspace(0, 1, 20)
            pnt = pnt.reshape(2, 4, 2).transpose(0, 2, 1).reshape(4, 4)
            points = np.outer((1 - u) ** 3, pnt[:, 0]) \
                + np.outer(3 * u * ((1 - u) ** 2), pnt[:, 1]) \
                + np.outer(3 * (u ** 2) * (1 - u), pnt[:, 2]) \
                + np.outer(u ** 3, pnt[:, 3])
            points = np.concatenate((points[:, :2], points[:, 2:]), axis=0)

        return points

    def _decode_recognition(self, rec):
        """解码识别结果 - 支持中文"""
        s = ''
        for c in rec:
            c = int(c)
            if c < self.voc_size - 1:
                s += self.CTLABELS[c]
            elif c == self.voc_size - 1:
                s += u'口'
        return s

    def _ctc_decode_recognition(self, rec):
        """CTC解码识别结果 - 支持中文"""
        last_char = False
        s = ''
        for c in rec:
            c = int(c)
            if c < self.voc_size - 1:
                if last_char != c:
                    s += self.CTLABELS[c]
                    last_char = c
            elif c == self.voc_size - 1:
                s += u'口'
            else:
                last_char = False
        return s

    def overlay_instances(self, ctrl_pnts, recs, scores, alpha=0.5):
        color = (0.1, 0.2, 0.5)

        for ctrl_pnt, rec, score in zip(ctrl_pnts, recs, scores):
            polygon = self._ctrl_pnt_to_poly(ctrl_pnt)
            self.draw_polygon(polygon, color, alpha=alpha)

            # draw text in the top left corner
            text = self._decode_recognition(rec)
            text = "{:.3f}: {}".format(score, text)
            lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
            text_pos = polygon[0]
            horiz_align = "left"
            font_size = self._default_font_size

            self.draw_text(
                text,
                text_pos,
                color=lighter_color,
                horizontal_alignment=horiz_align,
                font_size=font_size,
                draw_chinese=False if self.voc_size == 96 else True
            )
    

    def draw_text(
        self,
        text,
        position,
        *,
        font_size=None,
        color="g",
        horizontal_alignment="center",
        rotation=0,
        draw_chinese=False
    ):
        """
        Args:
            text (str): class label
            position (tuple): a tuple of the x and y coordinates to place text on image.
            font_size (int, optional): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            color: color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            horizontal_alignment (str): see `matplotlib.text.Text`
            rotation: rotation angle in degrees CCW
        Returns:
            output (VisImage): image object with text drawn.
        """
        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))
        
        x, y = position
        if draw_chinese:
            font_path = "./simsun.ttc"
            prop = mfm.FontProperties(fname=font_path)
            self.output.ax.text(
                x,
                y,
                text,
                size=font_size * self.output.scale,
                family="sans-serif",
                bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
                verticalalignment="top",
                horizontalalignment=horizontal_alignment,
                color=color,
                zorder=10,
                rotation=rotation,
                fontproperties=prop
            )
        else:
            self.output.ax.text(
                x,
                y,
                text,
                size=font_size * self.output.scale,
                family="sans-serif",
                bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
                verticalalignment="top",
                horizontalalignment=horizontal_alignment,
                color=color,
                zorder=10,
                rotation=rotation,
            )
        return self.output