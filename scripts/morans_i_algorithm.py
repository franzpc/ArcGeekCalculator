from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (QgsProcessingAlgorithm, QgsProcessingParameterVectorLayer,
                       QgsProcessingParameterField, QgsProcessingParameterNumber,
                       QgsProcessingParameterEnum, QgsProcessingOutputNumber,
                       QgsFeatureRequest, QgsVectorLayer, QgsFeature,
                       QgsProcessingParameterFeatureSink, QgsGeometry, QgsPointXY,
                       QgsWkbTypes, QgsFeatureSink, QgsField, QgsFields,
                       QgsProcessingUtils, QgsProcessingException, QgsProcessing)
import math
import numpy as np

class MoransIAlgorithm(QgsProcessingAlgorithm):
    """
    This algorithm calculates Moran's I spatial autocorrelation statistic 
    for a field in a point or polygon layer.
    """
    
    # Define constants for parameter names
    INPUT = 'INPUT'
    INPUT_FIELD = 'INPUT_FIELD'
    WEIGHT_TYPE = 'WEIGHT_TYPE'
    DISTANCE_BAND = 'DISTANCE_BAND'
    K_NEIGHBORS = 'K_NEIGHBORS'
    OUTPUT = 'OUTPUT'
    OUTPUT_MORANS_I = 'OUTPUT_MORANS_I'
    OUTPUT_Z_SCORE = 'OUTPUT_Z_SCORE'
    OUTPUT_P_VALUE = 'OUTPUT_P_VALUE'
    
    def __init__(self):
        super().__init__()
        # Check if numpy is available
        try:
            import numpy as np
            self.numpy_available = True
        except ImportError:
            self.numpy_available = False
            
        # Check if scipy is available
        try:
            from scipy import stats
            self.scipy_available = True
        except ImportError:
            self.scipy_available = False

    def initAlgorithm(self, config=None):
        """
        Define the inputs and outputs of the algorithm.
        """
        # Input layer
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT,
                self.tr('Input layer'),
                [QgsProcessing.TypeVectorAnyGeometry]
            )
        )
        
        # Field for analysis
        self.addParameter(
            QgsProcessingParameterField(
                self.INPUT_FIELD,
                self.tr('Field to analyze'),
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric
            )
        )
        
        # Weight type
        self.addParameter(
            QgsProcessingParameterEnum(
                self.WEIGHT_TYPE,
                self.tr('Spatial weights type'),
                options=['Distance band', 'K-nearest neighbors', 'Queen contiguity (polygons only)'],
                defaultValue=0
            )
        )
        
        # Distance band parameter
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DISTANCE_BAND,
                self.tr('Distance band (for distance-based weights)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1000.0,
                optional=True,
                minValue=0.0
            )
        )
        
        # K parameter
        self.addParameter(
            QgsProcessingParameterNumber(
                self.K_NEIGHBORS,
                self.tr('Number of neighbors (K) for K-nearest neighbors'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=8,
                optional=True,
                minValue=1
            )
        )
        
        # Output layer with Moran's I for each feature
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                self.tr('Output layer with local Moran\'s I'),
                QgsProcessing.TypeVector,
                optional=True,
                createByDefault=True
            )
        )
        
        # Add numeric outputs for Moran's I, z-score, and p-value
        self.addOutput(
            QgsProcessingOutputNumber(
                self.OUTPUT_MORANS_I,
                self.tr('Moran\'s I index')
            )
        )
        
        self.addOutput(
            QgsProcessingOutputNumber(
                self.OUTPUT_Z_SCORE,
                self.tr('Z-score')
            )
        )
        
        self.addOutput(
            QgsProcessingOutputNumber(
                self.OUTPUT_P_VALUE,
                self.tr('P-value')
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        """
        Process the algorithm.
        """
        # Check if required packages are available
        if not self.numpy_available:
            feedback.reportError(self.tr("This algorithm requires the numpy package. Please install it using pip: pip install numpy"))
            raise QgsProcessingException(self.tr("Missing dependency: numpy"))
            
        if not self.scipy_available:
            feedback.reportError(self.tr("This algorithm requires the scipy package. Please install it using pip: pip install scipy"))
            raise QgsProcessingException(self.tr("Missing dependency: scipy"))
        
        from scipy import stats
        
        # Get input parameters
        source = self.parameterAsVectorLayer(parameters, self.INPUT, context)
        field_name = self.parameterAsString(parameters, self.INPUT_FIELD, context)
        weight_type = self.parameterAsEnum(parameters, self.WEIGHT_TYPE, context)
        distance_band = self.parameterAsDouble(parameters, self.DISTANCE_BAND, context)
        k_neighbors = self.parameterAsInt(parameters, self.K_NEIGHBORS, context)
        
        # Validate input parameters
        if weight_type == 0 and (distance_band is None or distance_band <= 0):
            raise QgsProcessingException(self.tr("Distance band must be positive for distance-based weights"))
        
        if weight_type == 1 and (k_neighbors is None or k_neighbors <= 0):
            raise QgsProcessingException(self.tr("Number of neighbors (K) must be positive for K-nearest neighbors"))
        
        if weight_type == 2 and source.geometryType() != QgsWkbTypes.PolygonGeometry:
            raise QgsProcessingException(self.tr("Queen contiguity is only available for polygon layers"))
        
        # Get the field index
        field_index = source.fields().indexFromName(field_name)
        if field_index < 0:
            raise QgsProcessingException(self.tr(f"Field '{field_name}' not found"))
        
        # Prepare output fields
        fields = source.fields()
        fields.append(QgsField('morans_i', QVariant.Double))
        fields.append(QgsField('z_score', QVariant.Double))
        fields.append(QgsField('p_value', QVariant.Double))
        fields.append(QgsField('significance', QVariant.String))
        
        # Create output layer
        (sink, dest_id) = self.parameterAsSink(
            parameters, self.OUTPUT, context,
            fields, source.wkbType(), source.crs()
        )
        
        # Load all features and their values for the analysis
        features = list(source.getFeatures())
        feature_count = len(features)
        
        if feature_count < 3:
            feedback.reportError(self.tr("At least 3 features are required for Moran's I calculation"))
            return {
                self.OUTPUT: None,
                self.OUTPUT_MORANS_I: None,
                self.OUTPUT_Z_SCORE: None,
                self.OUTPUT_P_VALUE: None
            }
        
        # Extract values and positions
        values = []
        centroids = []
        valid_features = []
        
        for feature in features:
            # Get the value of the field
            value = feature[field_index]
            if value is None or (isinstance(value, (float)) and math.isnan(value)):
                feedback.pushWarning(self.tr(f"Feature {feature.id()} has NULL or NaN value and will be excluded"))
                continue
            
            values.append(float(value))
            centroids.append(feature.geometry().centroid().asPoint())
            valid_features.append(feature)
        
        # Verify we have enough valid values
        if len(values) < 3:
            feedback.reportError(self.tr("Not enough valid values for analysis"))
            return {
                self.OUTPUT: None,
                self.OUTPUT_MORANS_I: None,
                self.OUTPUT_Z_SCORE: None,
                self.OUTPUT_P_VALUE: None
            }
        
        # Convert to numpy array
        y = np.array(values)
        
        # Calculate weights matrix based on selected type
        feedback.pushInfo(self.tr(f"Creating spatial weights matrix..."))
        
        # Standardize y to z-scores
        y_mean = np.mean(y)
        y_std = np.std(y, ddof=1)  # Use sample standard deviation
        z = (y - y_mean) / y_std if y_std > 0 else np.zeros_like(y)
        
        # Initialize weights matrix
        W = np.zeros((len(centroids), len(centroids)))
        
        # Weight type name for reporting
        weight_type_names = ['Distance band', 'K-nearest neighbors', 'Queen contiguity']
        feedback.pushInfo(self.tr(f"Using {weight_type_names[weight_type]} weights"))
        
        # Compute weights based on the selected type
        if weight_type == 0:  # Distance band
            feedback.pushInfo(self.tr(f"Distance band: {distance_band}"))
            for i in range(len(centroids)):
                for j in range(len(centroids)):
                    if i == j:
                        continue
                    
                    # Calculate Euclidean distance
                    dist = math.sqrt((centroids[i].x() - centroids[j].x())**2 +
                                     (centroids[i].y() - centroids[j].y())**2)
                    
                    # Binary weight based on distance band
                    if dist <= distance_band:
                        W[i, j] = 1
        
        elif weight_type == 1:  # K-nearest neighbors
            feedback.pushInfo(self.tr(f"K-nearest neighbors: K={k_neighbors}"))
            
            # For each point, find K nearest neighbors
            for i in range(len(centroids)):
                # Calculate distances to all other points
                distances = []
                for j in range(len(centroids)):
                    if i == j:
                        distances.append((j, float('inf')))  # Exclude self
                    else:
                        dist = math.sqrt((centroids[i].x() - centroids[j].x())**2 +
                                        (centroids[i].y() - centroids[j].y())**2)
                        distances.append((j, dist))
                
                # Sort by distance and take K nearest
                distances.sort(key=lambda x: x[1])
                for j, _ in distances[:k_neighbors]:
                    W[i, j] = 1
        
        elif weight_type == 2:  # Queen contiguity
            feedback.pushInfo(self.tr("Using Queen contiguity weights"))
            
            # For polygons, check if they share any boundary or vertex
            for i, feature_i in enumerate(valid_features):
                for j, feature_j in enumerate(valid_features):
                    if i == j:
                        continue
                    
                    # Check if polygons are touching (Queen contiguity)
                    if feature_i.geometry().touches(feature_j.geometry()):
                        W[i, j] = 1
        
        # Row standardize weights
        row_sums = W.sum(axis=1)
        for i in range(len(W)):
            if row_sums[i] > 0:
                W[i, :] = W[i, :] / row_sums[i]
        
        # Calculate global Moran's I
        feedback.pushInfo(self.tr("Calculating global Moran's I..."))
        
        # Calculate numerator
        numerator = 0
        for i in range(len(z)):
            for j in range(len(z)):
                numerator += W[i, j] * z[i] * z[j]
        
        # Calculate denominator (sum of squares)
        denominator = np.sum(z**2)
        
        # Final Moran's I calculation
        n = len(z)
        s0 = np.sum(W)  # Sum of all weights
        
        if s0 == 0:
            feedback.reportError(self.tr("No spatial relationships found! Check your weight matrix parameters."))
            morans_i = None
            z_score = None
            p_value = None
        else:
            morans_i = (n / s0) * (numerator / denominator)
            
            # Calculate expected value E[I]
            expected_i = -1.0 / (n - 1)
            
            # Calculate variance of I
            s1 = 0.5 * np.sum((W + W.T)**2)
            s2 = np.sum((np.sum(W, axis=1) + np.sum(W, axis=0))**2)
            
            var_i = (
                (n**2 * s1 - n * s2 + 3 * s0**2) / 
                ((n**2 - 1) * s0**2)
            ) - expected_i**2
            
            # Calculate z-score and p-value
            z_score = (morans_i - expected_i) / math.sqrt(var_i)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test
        
        # Calculate local Moran's I for each feature
        feedback.pushInfo(self.tr("Calculating local Moran's I..."))
        
        # Significance classification
        def classify_significance(p):
            if p <= 0.001:
                return "p ≤ 0.001 (99.9%)"
            elif p <= 0.01:
                return "p ≤ 0.01 (99%)"
            elif p <= 0.05:
                return "p ≤ 0.05 (95%)"
            else:
                return "Not significant"
        
        # For each feature, calculate local Moran's I
        local_morans = []
        for i in range(len(valid_features)):
            # Calculate local Moran's I
            local_i = z[i] * np.sum(W[i, :] * z)
            
            # Theoretical mean and variance for local Moran's I
            local_mean = -W[i, :].sum() / (n - 1)
            
            # Add local Moran's I to feature attributes
            feat = QgsFeature(fields)
            # Copy original attributes
            for field_idx in range(len(source.fields())):
                feat.setAttribute(field_idx, valid_features[i][field_idx])
            
            # Add Moran's I statistics
            feat.setAttribute('morans_i', local_i)
            
            # Simplified for local measures (approximate)
            local_var = 1  # Simplified variance estimate
            local_z = (local_i - local_mean) / math.sqrt(local_var)
            local_p = 2 * (1 - stats.norm.cdf(abs(local_z)))
            
            feat.setAttribute('z_score', local_z)
            feat.setAttribute('p_value', local_p)
            feat.setAttribute('significance', classify_significance(local_p))
            
            # Set the geometry
            feat.setGeometry(valid_features[i].geometry())
            
            # Add to sink
            sink.addFeature(feat, QgsFeatureSink.FastInsert)
            
            local_morans.append(local_i)
        
        # Report global Moran's I results
        if morans_i is not None:
            feedback.pushInfo(self.tr(f"========== RESULTS =========="))
            feedback.pushInfo(self.tr(f"Global Moran's I Index: {morans_i:.6f}"))
            feedback.pushInfo(self.tr(f"Z-Score: {z_score:.6f}"))
            feedback.pushInfo(self.tr(f"P-value: {p_value:.6f}"))
            
            # Interpretation
            if p_value <= 0.05:
                if morans_i > expected_i:
                    feedback.pushInfo(self.tr(f"Interpretation: Significant positive spatial autocorrelation (clustered pattern)"))
                else:
                    feedback.pushInfo(self.tr(f"Interpretation: Significant negative spatial autocorrelation (dispersed pattern)"))
            else:
                feedback.pushInfo(self.tr(f"Interpretation: No significant spatial autocorrelation (random pattern)"))
            
            # Additional report with classification
            if p_value <= 0.001:
                significance = "p ≤ 0.001 (99.9% confidence level)"
            elif p_value <= 0.01:
                significance = "p ≤ 0.01 (99% confidence level)"
            elif p_value <= 0.05:
                significance = "p ≤ 0.05 (95% confidence level)"
            else:
                significance = "Not significant (random pattern)"
            
            feedback.pushInfo(self.tr(f"Significance: {significance}"))
            
            # Effect size description
            if abs(morans_i) <= 0.2:
                effect = "Weak"
            elif abs(morans_i) <= 0.6:
                effect = "Moderate"
            else:
                effect = "Strong"
            
            feedback.pushInfo(self.tr(f"Effect Size: {effect} spatial autocorrelation"))
            feedback.pushInfo(self.tr(f"============================"))
        else:
            feedback.pushInfo(self.tr(f"No results available due to insufficient spatial relationships"))
        
        return {
            self.OUTPUT: dest_id,
            self.OUTPUT_MORANS_I: morans_i,
            self.OUTPUT_Z_SCORE: z_score,
            self.OUTPUT_P_VALUE: p_value
        }

    def name(self):
        return 'moransispatialautocorrelation'

    def displayName(self):
        return self.tr('Moran\'s I Spatial Autocorrelation')

    def group(self):
        return self.tr('ArcGeek Calculator')

    def groupId(self):
        return 'arcgeekcalculator'

    def shortHelpString(self):
        return self.tr("""
        This algorithm calculates Moran's I spatial autocorrelation index for a feature layer.
        
        Moran's I measures spatial autocorrelation (feature similarity) based on feature locations and values.
        
        Parameters:
        - Input layer: The layer containing features to analyze (points or polygons)
        - Field to analyze: Numeric field whose values will be used for autocorrelation analysis
        - Spatial weights type: Method to determine which features are considered neighbors
            * Distance band: Features within specified distance are neighbors
            * K-nearest neighbors: K closest features are neighbors
            * Queen contiguity: Features sharing a boundary or vertex are neighbors (polygons only)
        - Distance band: Maximum distance for features to be considered neighbors
        - Number of neighbors (K): Number of nearest neighbors to use
        
        Outputs:
        - Output layer with local Moran's I: Each feature contains its local Moran's I value
        - Moran's I index: Global Moran's I statistic (-1 to +1)
            * +1: Perfect positive spatial autocorrelation (similar values cluster)
            * 0: No spatial autocorrelation (random pattern)
            * -1: Perfect negative spatial autocorrelation (dissimilar values cluster)
        - Z-score: Standard score indicating statistical significance
        - P-value: Probability value for statistical significance test
        
        Requirements:
        - This tool requires the numpy and scipy Python packages.
        """)

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return MoransIAlgorithm()