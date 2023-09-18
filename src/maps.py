import os
import io
import json
import requests
import warnings
from PIL import Image
from typing import List
from datetime import datetime

import ee
import numpy as np
import ipyleaflet as ipyl
import folium
from folium.plugins import Draw
from shapely.geometry import Polygon


from data_params import GEEData

EE_TILES = '{tile_fetcher.url_format}'


class LeafletMap(ipyl.Map):
    """
    A custom Map class.

    Inherits from ipyl.Map class.
    """

    slugs = ['Landsat-8-Surface-Reflectance', 'Sentinel-2-Top-of-Atmosphere-Reflectance']

    def __init__(self, geometry: dict = None, center: List[float] = [28.4, -16.4], zoom: int = 10, **kwargs):
        """
        Constructor for MapGEE class.

        Parameters:
        center: list, default [28.3904, -16.4409]
            The current center of the map.
        zoom: int, default 10
            The current zoom value of the map.
        geometry: dict, default None
            The current zoom value of the map.
        **kwargs: Additional arguments that are passed to the parent constructor.
        """
        self.geometry = geometry
        if self.geometry:
            self.center = self.centroid
        else:
            self.center = center
        self.zoom = zoom
        
        super().__init__(
            basemap=ipyl.basemap_to_tiles(ipyl.basemaps.Esri.WorldImagery),
            center=tuple(self.center), zoom=self.zoom, **kwargs)

        self.add_draw_control()
        self.add_gee_layers()

    def add_draw_control(self):
        control = ipyl.LayersControl(position='topright')
        self.add_control(control)

        print('Draw a rectangle on map to select and area.')

        draw_control = ipyl.DrawControl()

        draw_control.rectangle = {
            "shapeOptions": {
                "color": "#2BA4A0",
                "fillOpacity": 0,
                "opacity": 1
            }
        }

        if self.geometry:
                self.geometry['features'][0]['properties'] = {'style': {'color': "#2BA4A0", 'opacity': 1, 'fillOpacity': 0}}
                geo_json = ipyl.GeoJSON(
                    data=self.geometry
                )
                self.add_layer(geo_json)

        else:
            feature_collection = {
                'type': 'FeatureCollection',
                'features': []
            }

            def handle_draw(self, action, geo_json):
                """Do something with the GeoJSON when it's drawn on the map"""    
                #feature_collection['features'].append(geo_json)
                if 'pane' in list(geo_json['properties']['style'].keys()):
                    feature_collection['features'] = []
                else:
                    feature_collection['features'] = [geo_json]

            draw_control.on_draw(handle_draw)

            self.add_control(draw_control)

            self.geometry = feature_collection

    def add_gee_layers(self):
        for slug in self.slugs:
            gee_data = GEEData(dataset=slug)
            composite = gee_data.composite('2016-01-01', '2016-12-31')

            mapid = composite.getMapId(gee_data.vizz_params_rgb)
            tiles_url = EE_TILES.format(**mapid)

            composite_layer = ipyl.TileLayer(url=tiles_url, name=slug)
            self.add_layer(composite_layer)

    @property
    def polygon(self):
        if not self.geometry['features']:
            warnings.warn("Rectangle hasn't been drawn yet. Polygon is not available.")
            return None

        coordinates = self.geometry['features'][0]['geometry']['coordinates']
        return Polygon(coordinates[0])

    @property
    def bbox(self):
        if not self.polygon:
            warnings.warn("Rectangle hasn't been drawn yet. Bounding box is not available.")
            return None
        
        return list(self.polygon.bounds)
    
    @property
    def centroid(self):
        if not self.geometry['features']:
            warnings.warn("Rectangle hasn't been drawn yet. Centroid is not available.")
            return None
        else:
            return [arr[0] for arr in self.polygon.centroid.xy][::-1]
        
    def create_input_image(self):
        """
        Select region on map and create input image.
        Parameters
        ----------
        """
        if self.geometry.get('features') == []:
            raise ValueError(f'A rectangle has not been drawn on the map.')

        gee_data = GEEData(dataset=self.slugs[0])
        composite = gee_data.composite('2016-01-01', '2016-12-31')
        visSave = gee_data.vizz_params_rgb
        scale = gee_data.scale
        url = composite.getThumbURL({**visSave,**{'scale': scale}, **{'region':self.bbox}})

        response = requests.get(url)
        array = np.array(Image.open(io.BytesIO(response.content))) 
        array = array.reshape((1,) + array.shape) 


        ## Display input image on map
        ## Save the NumPy array as an image
        #image = Image.fromarray(array[0,:,:,:])
        #current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        #image_filename = f"rgb_image_{current_time}.png"
        #image.save(image_filename)
        ## Add image as overlay
        #image_overlay = ipyl.ImageOverlay(
        #    url=image_filename,  
        #    bounds=((self.bbox[1], self.bbox[0]), (self.bbox[3], self.bbox[2]))
        #    )
        #
        #self.add_layer(image_overlay)

        return array
    

class FoliumMap(folium.Map):
    """
    A custom Map class that can display Google Earth Engine tiles.

    Inherits from folium.Map class.
    """

    slug = 'Landsat-8-Surface-Reflectance'

    def __init__(self, center: List[float] = [25.0, 55.0], zoom: int = 3, **kwargs):
        """
        Constructor for MapGEE class.

        Parameters:
        center: list, default [25.0, 55.0]
            The current center of the map.
        zoom: int, default 3
            The current zoom value of the map.
        **kwargs: Additional arguments that are passed to the parent constructor.
        """
        self.center = center
        self.zoom = zoom
        self.geometry = None
        super().__init__(location=self.center, zoom_start=self.zoom, control_scale=True,
                         attr='Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>', **kwargs)

        self.add_draw_control()
        self.add_gee_layer()
        self.add_layer_control()

    def add_draw_control(self):
        draw = Draw(
            export=False,
            position="topleft",
            draw_options={
                "polyline": False,
                "poly": False,
                "circle": False,
                "polygon": False,
                "marker": False,
                "circlemarker": False,
                "rectangle": True
            }
        )

        draw.add_to(self)

    def add_gee_layer(self):
        """
        Add GEE layer to map.

        Parameters:
        image (ee.Image): The Earth Engine image to display.
        sld_interval (str): SLD style of discrete intervals to apply to the image.
        name (str): lLayer name.
        """

        gee_data = GEEData(dataset=self.slug)
        composite = gee_data.composite('2016-01-01', '2016-12-31')

        mapid = composite.getMapId(gee_data.vizz_params_rgb)
        tiles_url = EE_TILES.format(**mapid)

        tile_layer = folium.TileLayer(
            tiles=tiles_url,
            name="Input data",
            attr=self.slug,
            overlay=True,
            control=True,
            opacity=1
        )

        tile_layer.add_to(self)

    def add_layer_control(self):
        control = folium.LayerControl(position='topright')

        control.add_to(self)

    def create_input_image(self, geojson):
        """
        Select region on map and create input image.
        Parameters
        ----------
        """

        coordinates = geojson['geometry']['coordinates']
        polygon =  Polygon(coordinates[0])
        bbox = list(polygon.bounds)


        gee_data = GEEData(dataset=self.slug)
        composite = gee_data.composite('2016-01-01', '2016-12-31')
        visSave = gee_data.vizz_params_rgb
        scale = gee_data.scale
        url = composite.getThumbURL({**visSave,**{'scale': scale}, **{'region':bbox}})

        response = requests.get(url)
        array = np.array(Image.open(io.BytesIO(response.content))) 
        array = array.reshape((1,) + array.shape) 

        return array
    

    def add_output_image(self, geojson, png_image_path):

        # Get bbox
        coordinates = geojson['geometry']['coordinates']
        polygon =  Polygon(coordinates[0])
        bbox = list(polygon.bounds)

        # Add the image overlay to the map using the bbox
        image_overlay = folium.raster_layers.ImageOverlay(
            image=png_image_path,
            name="Prediction",
            bounds=[[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
            opacity=1,  
            interactive=True,
        )

        # Add the image overlay to the map
        image_overlay.add_to(self)

        return self


