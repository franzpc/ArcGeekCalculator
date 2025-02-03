from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (QgsProcessing, QgsFeatureSink, QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource, QgsProcessingParameterFeatureSink,
                       QgsProcessingParameterPoint, QgsProcessingParameterEnum,
                       QgsFeature, QgsGeometry, QgsPointXY,
                       QgsFields, QgsField, QgsWkbTypes, QgsProcessingException, QgsProject, QgsVectorLayer)
from collections import defaultdict, deque
import math

class LinesToOrderedPointsAlgorithm(QgsProcessingAlgorithm):
    INPUT = 'INPUT'
    START_POINT = 'START_POINT'
    METHOD = 'METHOD'
    OUTPUT = 'OUTPUT'

    def name(self):
        return 'linestoorderedpoints'

    def displayName(self):
        return self.tr('Lines to Ordered Points')

    def group(self):
        return self.tr('ArcGeek Calculator')

    def groupId(self):
        return 'arcgeekcalculator'

    def shortHelpString(self):
        return self.tr("""
        This algorithm orders points extracted from a line network, starting from a specified point.
        It numbers each point once, following the connectivity of the lines using either DFS (Depth-First Search) or BFS (Breadth-First Search).

        Parameters:
        - Input line layer: The layer containing the network lines.
        - Select the Start Point: Click on the map to set the starting point for the numbering.
        - Method: Choose the numbering method (DFS or BFS).

        Outputs:
        - A point layer with each vertex from the input lines, including:
          * Order: Numbering from the start point outwards using the selected method.

        Note: The numbering starts exactly from the selected start point or the closest point if the selected point is not exactly on a vertex.
        """)

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                self.tr('Input line layer'),
                [QgsProcessing.TypeVectorLine]
            )
        )
        self.addParameter(
            QgsProcessingParameterPoint(
                self.START_POINT,
                self.tr('Select the Start Point')
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.METHOD,
                self.tr('Select the method'),
                options=['DFS (Depth-First Search)', 'BFS (Breadth-First Search)'],
                defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                self.tr('Ordered points')
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        start_point = self.parameterAsPoint(parameters, self.START_POINT, context)
        method = self.parameterAsEnum(parameters, self.METHOD, context)

        fields = QgsFields()
        fields.append(QgsField('Order', QVariant.Int))
        fields.append(QgsField('X', QVariant.Double))
        fields.append(QgsField('Y', QVariant.Double))

        (sink, dest_id) = self.parameterAsSink(parameters, self.OUTPUT, context,
                                               fields, QgsWkbTypes.Point, source.sourceCrs())

        if source is None or sink is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))

        # Split lines and create graph
        points = []
        graph = defaultdict(set)
        for f in source.getFeatures():
            geom = f.geometry()
            if geom.isMultipart():
                parts = geom.asMultiPolyline()
            else:
                parts = [geom.asPolyline()]
            for part in parts:
                for i in range(len(part) - 1):
                    start, end = part[i], part[i+1]
                    if start not in points:
                        points.append(start)
                    if end not in points:
                        points.append(end)
                    start_index = points.index(start)
                    end_index = points.index(end)
                    graph[start_index].add(end_index)
                    graph[end_index].add(start_index)

        # Find the closest point to the start point
        start_index = min(range(len(points)), key=lambda i: points[i].distance(start_point))

        # DFS method
        def dfs(node, order, visited):
            if node in visited:
                return order
            visited.add(node)
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromPointXY(points[node]))
            f.setAttributes([order, points[node].x(), points[node].y()])
            sink.addFeature(f, QgsFeatureSink.FastInsert)

            neighbors = sorted(graph[node] - visited, key=lambda n: points[node].distance(points[n]))
            for neighbor in neighbors:
                order = dfs(neighbor, order + 1, visited)
            return order

        # BFS method
        def bfs(start_node):
            visited = set()
            queue = deque([(start_node, 1)])
            while queue:
                node, order = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                f = QgsFeature()
                f.setGeometry(QgsGeometry.fromPointXY(points[node]))
                f.setAttributes([order, points[node].x(), points[node].y()])
                sink.addFeature(f, QgsFeatureSink.FastInsert)

                neighbors = sorted(graph[node] - visited, key=lambda n: len(graph[n]))
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append((neighbor, order + 1))

        # Execute the selected method
        if method == 0:
            # DFS (Depth-First Search)
            visited = set()
            dfs(start_index, 1, visited)
        else:
            # BFS (Breadth-First Search)
            bfs(start_index)

        return {self.OUTPUT: dest_id}

    def createInstance(self):
        return LinesToOrderedPointsAlgorithm()
