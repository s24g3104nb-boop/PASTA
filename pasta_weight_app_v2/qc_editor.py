"""元画素座標のブラシと、承認状態を引き継がないUndo/Redo。"""
from __future__ import annotations
import base64
import hashlib
import io
from pathlib import Path
import cv2
from PIL import Image

try:
    from .photo_qc import _mask,_integer,apply_mask_edits
except ImportError:
    from photo_qc import _mask,_integer,apply_mask_edits


def brush_edit(mask, points, radius, *, erase=False, excluded_rects=()):
    mask=_mask(mask);h,w=mask.shape
    radius=_integer(radius,'ブラシ半径',1,100)
    if not isinstance(points,list) or not 1<=len(points)<=5001 or not isinstance(erase,bool):
        raise ValueError('ブラシの点列・操作が不正です。')
    coords=[]
    for p in points:
        if not isinstance(p,list) or len(p)!=2:raise ValueError('ブラシの座標は[x,y]です。')
        coords.append((_integer(p[0],'x',0,w-1),_integer(p[1],'y',0,h-1)))
    result=mask.copy();color=0 if erase else 255
    for p in coords:cv2.circle(result,p,radius,color,-1)
    for a,b in zip(coords,coords[1:]):cv2.line(result,a,b,color,2*radius+1)
    return apply_mask_edits(result,excluded_rects=excluded_rects)


def put_candidate(state,key,candidate):
    old=state.get(key)
    if old is not None:
        state[key+'_undo']=(state.get(key+'_undo',[])+[old])[-6:]
    state[key+'_redo']=[];state[key]=candidate


def history_step(state,key,direction):
    if direction not in ('undo','redo'):raise ValueError('Undo/Redo操作を指定してください。')
    source=state.get(key+'_'+direction,[])
    if not source:return False
    dest='redo' if direction=='undo' else 'undo'
    state[key+'_'+dest]=(state.get(key+'_'+dest,[])+[state[key]])[-6:]
    state[key]=source[-1];state[key+'_'+direction]=source[:-1]
    return True


def brush_component(rgb,mask,*,key):
    import streamlit.components.v1 as components
    component=components.declare_component('pasta_mask_brush',path=str(Path(__file__).parent/'components/mask_brush'))
    def encoded(array):
        f=io.BytesIO();Image.fromarray(array).save(f,'PNG')
        return 'data:image/png;base64,'+base64.b64encode(f.getvalue()).decode('ascii')
    revision=hashlib.sha256(mask.tobytes()).hexdigest()
    return component(image=encoded(rgb),mask=encoded(mask),revision=revision,key=key,default=None),revision
