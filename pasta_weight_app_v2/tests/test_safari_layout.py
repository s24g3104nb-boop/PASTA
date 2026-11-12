"""Contrast acceptance and overlap geometry from the reported mobile failures."""
import importlib.util
from pathlib import Path
import re

spec=importlib.util.spec_from_file_location('check_safari_layout',Path(__file__).parents[2]/'tools/check_safari_layout.py')
layout=importlib.util.module_from_spec(spec);spec.loader.exec_module(layout)


def rgb(hexcolor):
    return tuple(int(hexcolor[i:i+2],16) for i in (1,3,5))


def test_alert_markdown_contrast_in_both_palettes():
    # These are the actual pre-fix foreground and nested severity-surface colors.
    assert layout.contrast_ratio((242,243,238),(255,250,229))<1.2
    css=(Path(__file__).parents[1]/'ui/design.css').read_text()
    palettes=[dict(re.findall(r'(--pl-alert-[a-z-]+):\s*(#[0-9a-f]{6})',block))
              for block in re.findall(r'\{([^{}]*)\}',css)]
    palettes=[p for p in palettes if '--pl-alert-text' in p]
    assert len(palettes)>=2
    for palette in palettes:
        assert layout.contrast_ratio(rgb(palette['--pl-alert-text']),rgb(palette['--pl-alert-bg']))>=4.5
        assert layout.contrast_ratio(rgb(palette['--pl-alert-link']),rgb(palette['--pl-alert-bg']))>=4.5
        assert layout.contrast_ratio(rgb(palette['--pl-alert-text']),rgb(palette['--pl-alert-code-bg']))>=4.5


def test_viewport_overlap_detects_occluded_controls_and_not_touching_edges():
    nav=dict(x=0,y=750,width=390,height=94)
    assert layout.rectangles_overlap(dict(x=16,y=740,width=358,height=48),nav)
    assert not layout.rectangles_overlap(dict(x=16,y=702,width=358,height=48),nav)
    assert not layout.rectangles_overlap(dict(x=16,y=844,width=358,height=48),nav)
