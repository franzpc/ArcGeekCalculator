from qgis.PyQt.QtWidgets import QDialog, QVBoxLayout, QCheckBox, QPushButton, QLabel, QMessageBox
from qgis.PyQt.QtCore import QSettings
from qgis.core import QgsProject
from qgis.utils import iface

class BasemapManager(QDialog):
    def __init__(self, iface):
        super().__init__()
        self.iface = iface
        self.setWindowTitle("Manage Basemaps")
        self.layout = QVBoxLayout()
        
        self.basemaps = {
            'Google Satellite': ['connections-xyz', 'Google Satellite', '', '', 'Google', 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', '', '30', '0'],
            'Google Maps': ['connections-xyz', 'Google Maps', '', '', 'Google', 'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}', '', '30', '0'],
            'Google Terrain': ['connections-xyz', 'Google Terrain', '', '', 'Google', 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}', '', '30', '0'],
            'Bing VirtualEarth': ['connections-xyz', 'Bing VirtualEarth', '', '', 'Microsoft', 'http://ecn.t3.tiles.virtualearth.net/tiles/a{q}.jpeg?g=1', '', '30', '1'],
            'Esri Imagery': ['connections-xyz', 'Esri.WorldImagery', '', '', 'Tiles © Esri — Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community', 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', '', '30', '0'],
            'CartoDB Dark Matter': ['connections-xyz', 'CartoDB.DarkMatter', '', '', '© OpenStreetMap contributors © CARTO', 'https://cartodb-basemaps-a.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png', '', '30', '0'],
            'Esri World Topo Map': ['connections-xyz', 'Esri.WorldTopoMap', '', '', 'Tiles © Esri — Esri, DeLorme, NAVTEQ, TomTom, Intermap, iPC, USGS, FAO, NPS, NRCAN, GeoBase, Kadaster NL, Ordnance Survey, Esri Japan, METI, Esri China (Hong Kong), and the GIS User Community', 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}', '', '30', '0'],
            'OpenTopoMap': ['connections-xyz', 'OpenTopoMap', '', '', 'Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap (CC-BY-SA)', 'https://a.tile.opentopomap.org/{z}/{x}/{y}.png', '', '30', '0'],
            'USGS Topo': ['connections-xyz', 'USGS.USTopo', '', '', 'Tiles courtesy of the U.S. Geological Survey', 'https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}', '', '30', '0'],
            'USGS Imagery': ['connections-xyz', 'USGS.USImagery', '', '', 'Tiles courtesy of the U.S. Geological Survey', 'https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}', '', '30', '0'],
            'Esri Ocean Basemap': ['connections-xyz', 'Esri.OceanBasemap', '', '', 'Tiles © Esri — Sources: GEBCO, NOAA, CHS, OSU, UNH, CSUMB, National Geographic, DeLorme, NAVTEQ, and Esri', 'https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}', '', '30', '0'],
            'Esri National Geographic': ['connections-xyz', 'Esri.NatGeoWorldMap', '', '', 'Tiles © Esri — National Geographic, Esri, DeLorme, NAVTEQ, UNEP-WCMC, USGS, NASA, ESA, METI, NRCAN, GEBCO, NOAA, iPC', 'https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}', '', '30', '0'],
            'OpenStreetMap': ['connections-xyz', 'OpenStreetMap', '', '', 'OpenStreetMap', 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', '', '30', '0'],
            'NASA Shaded Relief': ['connections-xyz', 'NASA Shaded Relief', '', '', 'Imagery provided by services from the Global Imagery Browse Services (GIBS), operated by the NASA/GSFC/Earth Science Data and Information System with funding provided by NASA/HQ', 'https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/ASTER_GDEM_Greyscale_Shaded_Relief/default/GoogleMapsCompatible_Level12/{z}/{y}/{x}.jpg', '', '30', '0'],
        }

        self.checkboxes = {}
        
        for name, details in self.basemaps.items():
            checkbox = QCheckBox(name)
            checkbox.setChecked(name in ['Google Satellite', 'Bing VirtualEarth', 'Esri Imagery'])
            self.checkboxes[name] = checkbox
            self.layout.addWidget(checkbox)
        
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_basemaps)
        self.layout.addWidget(self.apply_button)
        
        self.setLayout(self.layout)

    def apply_basemaps(self):
        changes_made = False
        for name, checkbox in self.checkboxes.items():
            if checkbox.isChecked():
                if self.add_basemap(name, self.basemaps[name]):
                    changes_made = True
        
        if changes_made:
            # Reload connections
            iface.reloadConnections()
            QMessageBox.information(self, "Restart Required", 
                "Basemaps have been added successfully. Please restart QGIS to see the changes.")
        else:
            QMessageBox.information(self, "No Changes", 
                "No new basemaps were added.")
        
        self.close()

    def add_basemap(self, name, source):
        connectionType = source[0]
        connectionName = source[1]

        # Remove existing connection if it exists
        settings = QSettings()
        settings.beginGroup(f"qgis/{connectionType}")
        if connectionName in settings.childGroups():
            settings.beginGroup(connectionName)
            settings.remove("")  # This removes all keys under the group
            settings.endGroup()
        settings.endGroup()

        # Add new connection
        QSettings().setValue(f"qgis/{connectionType}/{connectionName}/authcfg", source[2])
        QSettings().setValue(f"qgis/{connectionType}/{connectionName}/password", source[3])
        QSettings().setValue(f"qgis/{connectionType}/{connectionName}/referer", source[4])
        QSettings().setValue(f"qgis/{connectionType}/{connectionName}/url", source[5])
        QSettings().setValue(f"qgis/{connectionType}/{connectionName}/username", source[6])
        QSettings().setValue(f"qgis/{connectionType}/{connectionName}/zmax", source[7])
        QSettings().setValue(f"qgis/{connectionType}/{connectionName}/zmin", source[8])
        return True