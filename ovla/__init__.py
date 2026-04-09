"""
O-VLA: Universal Vision-Language-Action Transfer

A middleware system enabling ANY VLA model to control ANY robot.
"""

__version__ = "0.1.0"
__author__ = "Ansh Bhansali"
__email__ = "anshb3@illinois.edu"

from ovla.core.pipeline import OVLAPipeline

__all__ = ["OVLAPipeline"]
