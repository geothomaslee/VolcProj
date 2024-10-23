#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 13:04:03 2024

@author: thomaslee
"""

@dataclass
class Animator:
    frames=None

    def __post_init__(self):
        self.frames = []

    def _save_figure_to_frame(self,fig):
        buf = io.BytesIO()
        fig.savefig(buf,format='png',bbox_inches='tight',dpi=100)
        buf.seek(0)
        return Image.open(buf)

    def _save_animation(self, path: str):
        self.frames[0].save(
            path,
            save_all=True,
            append_images=self.frames[1:],
            optimize=False,
            duration=50,
            loop=0)

    def close(self):
        self.frames = []