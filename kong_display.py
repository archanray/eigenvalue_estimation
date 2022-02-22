"""
code to display kong dataset
"""

from src.get_dataset import get_data
from src.display_codes import display_kong_dataset

xy, dataset_size = get_data("kong")
display_kong_dataset(xy)
