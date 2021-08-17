"""
Nmask Ancillary Summaries

TODO(pjt554): Description goes here

"""
import dask
from typing import Optional, Tuple
import numpy as np
import xarray as xr
from odc.stats.model import Task
from odc.algo.io import load_with_native_transform
from odc.algo import safe_div, apply_numexpr, keep_good_only, binary_dilation
from odc.algo.io import dc_load
from odc.stats.model import StatsPluginInterface
from odc.stats import _plugins

from . import nmask_pmod


class StatsNmask(StatsPluginInterface):
    """
    Generate Nmask ancillary summaries.
    """

    NAME = "ga_s2_nmask_ancillary"
    SHORT_NAME = NAME
    VERSION = "0.0.1"
    PRODUCT_FAMILY = "nmask_ancillary"

    def __init__(
        self,
        resampling: str = "bilinear",
    ):
        self.resampling = resampling

    def _native_tr(self, xx):
        return xx

    @property
    def measurements(self) -> Tuple[str, ...]:
        return ('s6m', 's6m_std',
                'mndwi', 'mndwi_std',
                'msavi', 'msavi_std',
                'whi', 'whi_std')

    def input_data(self, task: Task) -> xr.Dataset:
        chunks = {"y": -1, "x": -1}
        groupby = "solar_day"

        xx = load_with_native_transform(
            task.datasets,
            bands=["nbart_green", "nbart_blue", "nbart_red",
                   "nbart_nir_2", "nbart_swir_2", "nbart_swir_3"],
            geobox=task.geobox,
            groupby=groupby,
            resampling=self.resampling,
            chunks=chunks,
            native_transform=self._native_tr,
        )

        return xx

    def reduce(self, xx: xr.Dataset) -> xr.Dataset:
        template = xr.Dataset({m: xx.nbart_blue
                               for m in self.measurements})
        return xr.map_blocks(nmask_pmod.summarise, xx, template=template)

_plugins.register("nmask-ancillary", StatsNmask)
