from qgis.PyQt.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout
from qgis.gui import QgsMapLayerComboBox
from qgis.core import QgsMapLayerProxyModel

class ParcelSplitParametersDialog(QDialog):
    def __init__(self, alg, parent=None):
        super().__init__(parent)
        self.alg = alg
        self.setupUi()

    def setupUi(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Input layer selection
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel('Input polygon layer:'))
        self.input_layer = QgsMapLayerComboBox()
        self.input_layer.setFilters(QgsMapLayerProxyModel.PolygonLayer)
        input_layout.addWidget(self.input_layer)
        layout.addLayout(input_layout)

        # Split type
        split_layout = QHBoxLayout()
        split_layout.addWidget(QLabel('Split type:'))
        self.split_type_combo = QComboBox()
        self.split_type_combo.addItems(['By measures', 'Equal parts', 'By area'])
        self.split_type_combo.currentIndexChanged.connect(self.updateFieldVisibility)
        split_layout.addWidget(self.split_type_combo)
        layout.addLayout(split_layout)

        # By measures fields
        self.measures_widget = QWidget()
        measures_layout = QVBoxLayout(self.measures_widget)
        self.width_input = QLineEdit()
        self.length_input = QLineEdit()
        measures_layout.addWidget(QLabel('Desired lot width:'))
        measures_layout.addWidget(self.width_input)
        measures_layout.addWidget(QLabel('Desired lot length:'))
        measures_layout.addWidget(self.length_input)
        layout.addWidget(self.measures_widget)

        # Equal parts field
        self.parts_widget = QWidget()
        parts_layout = QVBoxLayout(self.parts_widget)
        self.num_parts_input = QLineEdit()
        parts_layout.addWidget(QLabel('Number of parts to split into:'))
        parts_layout.addWidget(self.num_parts_input)
        layout.addWidget(self.parts_widget)

        # By area field
        self.area_widget = QWidget()
        area_layout = QVBoxLayout(self.area_widget)
        self.target_area_input = QLineEdit()
        area_layout.addWidget(QLabel('Target area for each lot:'))
        area_layout.addWidget(self.target_area_input)
        layout.addWidget(self.area_widget)

        # Common fields
        self.min_area_ratio_input = QLineEdit()
        self.min_area_ratio_input.setText('0.05')
        layout.addWidget(QLabel('Minimum area ratio for annexation:'))
        layout.addWidget(self.min_area_ratio_input)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self.updateFieldVisibility(0)

    def updateFieldVisibility(self, index):
        self.measures_widget.setVisible(index == 0)
        self.parts_widget.setVisible(index == 1)
        self.area_widget.setVisible(index == 2)

    def getParameters(self):
        parameters = {}
        parameters[self.alg.INPUT] = self.input_layer.currentLayer().id() if self.input_layer.currentLayer() else None
        parameters[self.alg.SPLIT_TYPE] = self.split_type_combo.currentIndex()
        parameters[self.alg.WIDTH] = self.width_input.text()
        parameters[self.alg.LENGTH] = self.length_input.text()
        parameters[self.alg.NUM_PARTS] = self.num_parts_input.text()
        parameters[self.alg.TARGET_AREA] = self.target_area_input.text()
        parameters[self.alg.MIN_AREA_RATIO] = self.min_area_ratio_input.text()
        return parameters

    def setParameters(self, parameters):
        if self.alg.INPUT in parameters:
            self.input_layer.setLayer(parameters[self.alg.INPUT])
        if self.alg.SPLIT_TYPE in parameters:
            self.split_type_combo.setCurrentIndex(int(parameters[self.alg.SPLIT_TYPE]))
        if self.alg.WIDTH in parameters:
            self.width_input.setText(str(parameters[self.alg.WIDTH]))
        if self.alg.LENGTH in parameters:
            self.length_input.setText(str(parameters[self.alg.LENGTH]))
        if self.alg.NUM_PARTS in parameters:
            self.num_parts_input.setText(str(parameters[self.alg.NUM_PARTS]))
        if self.alg.TARGET_AREA in parameters:
            self.target_area_input.setText(str(parameters[self.alg.TARGET_AREA]))
        if self.alg.MIN_AREA_RATIO in parameters:
            self.min_area_ratio_input.setText(str(parameters[self.alg.MIN_AREA_RATIO]))

    def results(self):
        return self.getParameters()