import numpy as np
from osgeo import gdal
from qgis.core import (
    QgsCoordinateTransform,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFileDestination,
    QgsRaster,
)
from qgis.PyQt.QtCore import QCoreApplication
import os

# Check dependencies at module level
HAS_SKLEARN = True
HAS_NUMPY = True

try:
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.mixture import GaussianMixture
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    from sklearn.model_selection import train_test_split, cross_val_score
except ImportError:
    HAS_SKLEARN = False

try:
    import numpy
except ImportError:
    HAS_NUMPY = False

# Import the scientific report generator
from .classification_report_generator import generate_scientific_classification_report

class EnhancedClassificationAlgorithm(QgsProcessingAlgorithm):
    """
    This algorithm runs various machine learning classification methods on an image
    using training data specified in a point vector layer with scientific rigor.
    """

    TRAINING_DATA = "TRAINING_DATA"
    CLASSIFICATION_FIELD = "CLASSIFICATION_FIELD"
    SOURCE_IMAGE = "SOURCE_IMAGE"
    CLASSIFIED_IMAGE = "CLASSIFIED_IMAGE"
    ALGORITHM_TYPE = "ALGORITHM_TYPE"
    N_ESTIMATORS = "N_ESTIMATORS"
    TEST_SIZE = "TEST_SIZE"
    CROSS_VALIDATION = "CROSS_VALIDATION"
    GENERATE_REPORT = "GENERATE_REPORT"
    OUTPUT_REPORT = "OUTPUT_REPORT"

    def tr(self, string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):
        return EnhancedClassificationAlgorithm()

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm.
        """
        return "enhancedclassification"

    def displayName(self):
        """
        Returns the translated algorithm name.
        """
        return self.tr("Enhanced Image Classification")

    def group(self):
        """
        Returns the name of the group this algorithm belongs to.
        """
        return self.tr("ArcGeek Calculator")

    def groupId(self):
        """
        Returns the unique ID of the group this algorithm belongs to.
        """
        return "arcgeekcalculator"

    def shortHelpString(self):
        """
        Returns a localized short helper string for the algorithm.
        """
        return self.tr(
            """
            This algorithm performs advanced image classification using various machine learning methods 
            with scientifically rigorous accuracy assessment.
            
            Available algorithms:
            - Random Forest: Robust ensemble learning method with feature importance analysis
            - Gaussian Mixture Model: Probabilistic model for clustering
            - Support Vector Machine: Powerful discriminative classifier
            - K-Nearest Neighbors: Simple distance-based classifier
            
            The tool requires:
            - A point vector layer with class labels for training
            - The classification field containing class labels
            - A multiband raster image to classify
            - Algorithm selection and parameters
            
            Outputs:
            - Classified image with a raster attribute table
            - Optional comprehensive HTML report following remote sensing standards
            
          
            Note: This algorithm requires scikit-learn and numpy to be installed.
            For full report functionality, matplotlib and seaborn are also recommended.
            """
        )

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and output of the algorithm.
        """
        # Algorithm selector
        self.addParameter(
            QgsProcessingParameterEnum(
                self.ALGORITHM_TYPE,
                self.tr("Classification Algorithm"),
                options=["Random Forest", "Gaussian Mixture Model", "Support Vector Machine", "K-Nearest Neighbors"],
                defaultValue=0
            )
        )

        # Configurable parameters
        self.addParameter(
            QgsProcessingParameterNumber(
                self.N_ESTIMATORS,
                self.tr("Number of estimators (for Random Forest)"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=300,
                minValue=10,
                maxValue=1000
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.TEST_SIZE,
                self.tr("Test set size (proportion)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.33,
                minValue=0.1,
                maxValue=0.5
            )
        )
        
        # Cross-validation option
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.CROSS_VALIDATION,
                self.tr("Perform cross-validation assessment"),
                defaultValue=True
            )
        )

        # Input data
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.TRAINING_DATA,
                self.tr("Training data points"),
                [QgsProcessing.TypeVectorPoint],
            )
        )

        self.addParameter(
            QgsProcessingParameterField(
                self.CLASSIFICATION_FIELD,
                self.tr("Classification field"),
                parentLayerParameterName=self.TRAINING_DATA,
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.SOURCE_IMAGE, 
                self.tr("Multiband image to classify"), 
                [QgsProcessing.TypeRaster]
            )
        )

        # Option to generate HTML report
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.GENERATE_REPORT,
                self.tr("Generate classification report (HTML)"),
                defaultValue=False
            )
        )

        # Output HTML report
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_REPORT,
                self.tr("Classification report (HTML)"),
                fileFilter="HTML files (*.html)",
                optional=True
            )
        )

        # Output
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.CLASSIFIED_IMAGE, 
                self.tr("Classified Image")
            )
        )

    def checkParameterValues(self, parameters, context):
        """
        This function checks if required libraries are available.
        """
        # Check if required libraries are available
        missing_libs = []
        
        if not HAS_SKLEARN:
            missing_libs.append("scikit-learn")
            
        if not HAS_NUMPY:
            missing_libs.append("numpy")
        
        if missing_libs:
            return False, self.tr(f"This algorithm requires the following Python libraries: {', '.join(missing_libs)}. Please install them using pip: pip install {' '.join(missing_libs)}")
        
        # Check if report generation is requested and output path is provided
        generate_report = self.parameterAsBool(parameters, self.GENERATE_REPORT, context)
        if generate_report:
            output_report = self.parameterAsString(parameters, self.OUTPUT_REPORT, context)
            if not output_report:
                return False, self.tr("When 'Generate scientific classification report' is checked, a report output file must be specified.")
        
        return super().checkParameterValues(parameters, context)

    def perform_cross_validation(self, classifier, X, y, cv_folds=5, feedback=None):
        """
        Perform k-fold cross-validation for more robust accuracy assessment
        """
        try:
            if feedback:
                feedback.pushInfo(f"Performing {cv_folds}-fold cross-validation...")
            
            cv_scores = cross_val_score(classifier, X, y, cv=cv_folds, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            if feedback:
                feedback.pushInfo(f"Cross-validation results: {cv_mean:.4f} ± {cv_std:.4f}")
                feedback.pushInfo(f"Individual fold scores: {[f'{score:.4f}' for score in cv_scores]}")
            
            return cv_mean, cv_std, cv_scores
        except Exception as e:
            if feedback:
                feedback.pushWarning(f"Cross-validation failed: {str(e)}")
            return None, None, None

    def processAlgorithm(self, parameters, context, feedback):
        """
        Here is where the processing itself takes place.
        """
        # Import sklearn components here to ensure they're available
        if not HAS_SKLEARN:
            raise QgsProcessingException(self.tr("This algorithm requires scikit-learn. Please install it using pip install scikit-learn."))
        
        # Get parameters
        algorithm_type = self.parameterAsEnum(parameters, self.ALGORITHM_TYPE, context)
        n_estimators = self.parameterAsInt(parameters, self.N_ESTIMATORS, context)
        test_size = self.parameterAsDouble(parameters, self.TEST_SIZE, context)
        cross_validation = self.parameterAsBool(parameters, self.CROSS_VALIDATION, context)
        generate_report = self.parameterAsBool(parameters, self.GENERATE_REPORT, context)
        
        source = self.parameterAsSource(parameters, self.TRAINING_DATA, context)
        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.TRAINING_DATA))
        
        field_name = self.parameterAsString(parameters, self.CLASSIFICATION_FIELD, context)
        sourceImage = self.parameterAsRasterLayer(parameters, self.SOURCE_IMAGE, context)
        if sourceImage is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.SOURCE_IMAGE))

        # Extract data and preparation
        feedback.pushInfo("🔍 Extracting image properties...")
        extent = sourceImage.extent()
        xmin = extent.xMinimum()
        ymax = extent.yMaximum()
        crs = sourceImage.crs()
        provider = sourceImage.dataProvider()
        num_bands = provider.bandCount()
        
        # Check if the raster has bands
        if num_bands < 1:
            raise QgsProcessingException(self.tr("Invalid raster: no bands found"))
            
        feedback.pushInfo(f"📊 Image info: {sourceImage.width()}x{sourceImage.height()} pixels, {num_bands} bands")
        
        # Scientific validation: Check if multiband for meaningful classification
        if num_bands == 1:
            feedback.pushWarning("⚠️ Single-band image detected. For optimal classification results, use multiband imagery.")
        
        # Validate minimum sample size according to scientific standards
        min_recommended_samples = max(10 * num_bands, 30)
        total_features = source.featureCount()
        if total_features < min_recommended_samples:
            feedback.pushWarning(f"⚠️ Sample size warning: {total_features} training points available. "
                                f"Recommended minimum: {min_recommended_samples} points for {num_bands} bands.")

        label_field_index = source.fields().indexFromName(field_name)
        if label_field_index < 0:
            raise QgsProcessingException(f"No attribute named '{field_name}' in the training layer")

        # Coordinate transform between vector and raster
        transform = QgsCoordinateTransform(
            source.sourceCrs(), sourceImage.crs(), context.project()
        )

        feedback.pushInfo("📍 Collecting training data...")
        # Collection of training data
        classes = {}
        feature_list = []
        
        for feature in source.getFeatures():
            try:
                # Identify the raster values at the point for all bands
                point = transform.transform(feature.geometry().asPoint())
                results = provider.identify(point, QgsRaster.IdentifyFormatValue)

                # Each sample contains: all band values first and the label last
                if results.isValid():
                    class_title = feature.attributes()[label_field_index]
                    
                    # Skip features with NULL class values
                    if class_title is None:
                        feedback.pushWarning(f"Skipping point with NULL classification value at {point.toString()}")
                        continue
                        
                    # Add to class dictionary if new
                    if class_title not in classes:
                        classes[class_title] = len(classes)
                    class_id = classes[class_title]
                    
                    values = []
                    valid_point = True
                    
                    # Collect values from all bands
                    for band in range(1, num_bands + 1):
                        if band in results.results():
                            band_value = results.results()[band]
                            
                            # Skip if band value is None or NaN
                            if band_value is None or (isinstance(band_value, float) and np.isnan(band_value)):
                                valid_point = False
                                feedback.pushWarning(f"Skipping point with NULL band value at {point.toString()}")
                                break
                                
                            values.append(band_value)
                        else:
                            valid_point = False
                            feedback.pushWarning(f"Band {band} not found in raster results at {point.toString()}")
                            break
                    
                    # Only add if all bands have valid values
                    if valid_point and len(values) == num_bands:
                        values.append(class_id)
                        feature_list.append(values)
                    
                else:
                    feedback.pushWarning(
                        f"Could not identify raster values at point {feature.geometry().asWkt()}"
                    )
            except Exception as e:
                feedback.pushWarning(f"Error processing point {feature.id()}: {str(e)}")
                continue

        # Check if we have enough training data
        if len(feature_list) < 5:
            raise QgsProcessingException(self.tr("Not enough valid training points found. At least 5 points are required."))
            
        feedback.pushInfo(f"✅ Collected {len(feature_list)} training samples in {len(classes)} classes.")
        
        # Scientific validation: Check class balance
        unique_classes, class_counts = np.unique([f[-1] for f in feature_list], return_counts=True)
        min_class_samples = min(class_counts)
        if min_class_samples < max(10, num_bands):
            feedback.pushWarning(f"⚠️ Smallest class has only {min_class_samples} samples. "
                                f"Recommended minimum: {max(10, num_bands)} per class.")

        # Prepare data for training
        feedback.pushInfo("🔧 Preparing data for analysis...")
        feature_array = np.array(feature_list)
        pixelSizeX = sourceImage.rasterUnitsPerPixelX()
        pixelSizeY = sourceImage.rasterUnitsPerPixelY()
        width = sourceImage.width()
        height = sourceImage.height()
        
        # Extract raster data with progress updates
        feedback.pushInfo(f"📖 Reading {num_bands} raster bands...")
        try:
            # Use GDAL for reliable raster reading
            gdal_dataset = gdal.Open(sourceImage.source())
            if gdal_dataset is None:
                raise QgsProcessingException(self.tr("Could not open raster dataset with GDAL"))
                
            dataArray = []
            for i in range(1, num_bands + 1):
                if feedback.isCanceled():
                    return {}
                    
                feedback.setProgress(int((i-1) * 20 / num_bands))
                feedback.pushInfo(f"Reading band {i} of {num_bands}...")
                band = gdal_dataset.GetRasterBand(i)
                band_data = band.ReadAsArray()
                if band_data is None:
                    raise QgsProcessingException(self.tr(f"Failed to read band {i} - it may be too large or corrupted."))
                dataArray.append(band_data)
            
            # Convert to numpy array
            dataArray = np.array(dataArray)
            feedback.pushInfo(f"📊 Raster data shape: {dataArray.shape}")
            
        except Exception as e:
            feedback.reportError(f"Error reading raster data: {str(e)}")
            raise QgsProcessingException(self.tr("Failed to read raster data. Try using a smaller image or a different format."))
        
        # Validate raster shape
        if dataArray.size == 0:
            raise QgsProcessingException(self.tr("Empty raster data - no pixel values found."))
        
        # Get dimensions and reshape
        if len(dataArray.shape) == 3:
            bands, height, width = dataArray.shape
            feedback.pushInfo(f"📊 Multi-band image: {bands} bands, {height}x{width} pixels")
        else:
            # Handle single band images
            height, width = dataArray.shape
            bands = 1
            dataArray = dataArray.reshape(1, height, width)
            feedback.pushInfo(f"📊 Single-band image: {height}x{width} pixels")
        
        # Check if bands count matches
        if bands != num_bands:
            feedback.pushWarning(f"⚠️ Band count mismatch - metadata says {num_bands}, actual data has {bands} bands")
        
        feedback.pushInfo("🔄 Reshaping image data for analysis...")  
        try:
            # Reshape to have each pixel as a row and each band as a column
            reshaped_image = dataArray.reshape(bands, height * width).T
            feedback.pushInfo(f"📊 Reshaped data size: {reshaped_image.shape}")
        except Exception as e:
            feedback.reportError(f"Error reshaping data: {str(e)}")
            raise QgsProcessingException(self.tr(f"Failed to reshape raster data: {str(e)}"))

        # Prepare training and test sets
        feedback.pushInfo("📊 Preparing training and test data...")
        X = feature_array[:, 0:bands]
        y = feature_array[:, -1]
        
        feedback.pushInfo(f"📈 Training data: {X.shape}, labels: {y.shape}")
        feedback.pushInfo(f"📊 Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        feedback.pushInfo(f"🔀 Splitting data with test size: {test_size}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=7)
        feedback.pushInfo(f"📊 Training set: {X_train.shape}, Test set: {X_test.shape}")

        # Select and initialize the classifier
        feedback.setProgress(30)
        feedback.pushInfo("🤖 Training the classifier...")
        
        if algorithm_type == 0:  # Random Forest
            feedback.pushInfo(f"🌲 Using Random Forest Classifier with {n_estimators} estimators")
            classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=7)
            classifier.fit(X_train, y_train)
            y_ts = classifier.predict(X_test)
            
            # Feature importance analysis
            feature_importance = classifier.feature_importances_
            for i, importance in enumerate(feature_importance):
                feedback.pushInfo(f"📊 Band {i+1} importance: {importance:.4f}")
                
        elif algorithm_type == 1:  # Gaussian Mixture Model
            feedback.pushInfo("🔄 Using Gaussian Mixture Model")
            # GMM is a generative model that needs to be adapted for classification
            n_classes = len(classes)
            classifier = GaussianMixture(n_components=n_classes, random_state=7)
            # For GMM, training is different as it's an unsupervised model adapted to supervised
            classifier.fit(X_train)
            y_ts = classifier.predict(X_test)
            feature_importance = None
            
        elif algorithm_type == 2:  # Support Vector Machine
            feedback.pushInfo("⚡ Using Support Vector Machine")
            classifier = SVC(kernel='rbf', random_state=7)
            classifier.fit(X_train, y_train)
            y_ts = classifier.predict(X_test)
            feature_importance = None
            
        elif algorithm_type == 3:  # K-Nearest Neighbors
            feedback.pushInfo("🎯 Using K-Nearest Neighbors")
            classifier = KNeighborsClassifier(n_neighbors=5)
            classifier.fit(X_train, y_train)
            y_ts = classifier.predict(X_test)
            feature_importance = None

        feedback.setProgress(40)
        
        # Perform cross-validation if requested
        cv_mean, cv_std, cv_scores = None, None, None
        if cross_validation and algorithm_type != 1:  # Skip for GMM as it's unsupervised
            cv_mean, cv_std, cv_scores = self.perform_cross_validation(classifier, X, y, cv_folds=5, feedback=feedback)

        feedback.setProgress(50)
        feedback.pushInfo("📊 Calculating classification accuracy...")
        
        # Calculate accuracy
        OA = accuracy_score(y_test, y_ts)
        CFM = confusion_matrix(y_test, y_ts)
        feedback.pushInfo(f"🎯 Classification performance: Overall Accuracy = {OA:.4f} ({OA*100:.2f}%)")
        
        # Calculate scientific metrics
        try:
            from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score
            kappa = cohen_kappa_score(y_test, y_ts)
            balanced_acc = balanced_accuracy_score(y_test, y_ts)
            feedback.pushInfo(f"🔬 Cohen's Kappa = {kappa:.4f}")
            feedback.pushInfo(f"⚖️ Balanced Accuracy = {balanced_acc:.4f}")
            
            # Kappa interpretation
            if kappa >= 0.8:
                kappa_interp = "Almost Perfect"
            elif kappa >= 0.6:
                kappa_interp = "Substantial"
            elif kappa >= 0.4:
                kappa_interp = "Moderate"
            elif kappa >= 0.2:
                kappa_interp = "Fair"
            else:
                kappa_interp = "Poor"
            feedback.pushInfo(f"📈 Kappa interpretation: {kappa_interp}")
            
        except ImportError:
            feedback.pushWarning("⚠️ Advanced metrics unavailable - install full scikit-learn")

        # Apply classification to the entire image
        feedback.setProgress(60)
        feedback.pushInfo("🖼️ Classifying the full image...")
        
        try:
            classifiedImage = classifier.predict(reshaped_image)
            feedback.pushInfo(f"✅ Classification complete. Unique classes in result: {np.unique(classifiedImage)}")
        except Exception as e:
            feedback.reportError(f"Error during classification: {str(e)}")
            raise QgsProcessingException(self.tr(f"Classification failed: {str(e)}. The image may be too large or complex."))

        # Format the classified image for saving
        feedback.setProgress(70)
        feedback.pushInfo("💾 Formatting classified image...")
        try:
            classifiedImage_reshaped = classifiedImage.reshape(height, width)
            geotransform = (xmin, pixelSizeX, 0, ymax, 0, -pixelSizeY)
        except Exception as e:
            feedback.reportError(f"Error reshaping classification result: {str(e)}")
            raise QgsProcessingException(self.tr(f"Failed to format classification result: {str(e)}"))

        # Save the classified image
        feedback.setProgress(80)
        feedback.pushInfo("💾 Saving classified image...")
        output_file = self.parameterAsOutputLayer(parameters, self.CLASSIFIED_IMAGE, context)
        
        try:
            driver = gdal.GetDriverByName("GTiff")
            out_raster = driver.Create(output_file, width, height, 1, gdal.GDT_Float32)
            
            if out_raster is None:
                raise QgsProcessingException(self.tr(f"Failed to create output raster at {output_file}"))
                
            out_raster.SetGeoTransform(geotransform)
            out_raster.SetProjection(crs.toWkt())
            out_band = out_raster.GetRasterBand(1)
            out_band.WriteArray(classifiedImage_reshaped)
            
            # Create raster attribute table for the image
            feedback.pushInfo("📋 Creating raster attribute table...")
            rat = gdal.RasterAttributeTable()
            rat.CreateColumn("Value", gdal.GFT_Integer, gdal.GFU_MinMax)
            rat.CreateColumn("Class_name", gdal.GFT_String, gdal.GFU_Name)

            for class_title, class_id in classes.items():
                row_index = rat.GetRowCount()
                rat.SetRowCount(row_index + 1)
                rat.SetValueAsInt(row_index, 0, class_id)
                rat.SetValueAsString(row_index, 1, str(class_title))

            out_band.SetDefaultRAT(rat)
            out_band.FlushCache()
            out_raster = None  # Close the file
            
        except Exception as e:
            feedback.reportError(f"Error saving output raster: {str(e)}")
            raise QgsProcessingException(self.tr(f"Failed to save output raster: {str(e)}"))
        
        # Generate scientific HTML report if requested
        feedback.setProgress(90)
        output_report = None
        if generate_report:
            output_report = self.parameterAsString(parameters, self.OUTPUT_REPORT, context)
            if output_report:
                feedback.pushInfo(f"📊 Generating comprehensive scientific classification report: {output_report}")
                try:
                    # Count samples per class for distribution chart
                    unique_classes, class_counts = np.unique(y, return_counts=True)
                    
                    # Generate comprehensive scientific HTML report
                    html_content = generate_scientific_classification_report(
                        parameters, context, classifier, classes, 
                        X_test, y_test, class_counts, CFM, OA, 
                        feature_importance, algorithm_type, sourceImage, output_file, num_bands
                    )
                    
                    # Save HTML report
                    with open(output_report, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                        
                    feedback.pushInfo("📊 Comprehensive scientific classification report created successfully!")
                    
                    # Try to open the report in default browser
                    try:
                        import webbrowser
                        webbrowser.open('file://' + os.path.abspath(output_report))
                        feedback.pushInfo("🌐 Report opened in default web browser.")
                    except:
                        feedback.pushInfo("📄 Report saved successfully. You can open it manually in your web browser.")
                        
                except Exception as e:
                    feedback.pushWarning(f"⚠️ Error generating classification report: {str(e)}")
                    feedback.pushWarning("✅ Classification completed successfully, but the report could not be generated.")
            else:
                feedback.pushInfo("📊 Report generation was requested but no output path was specified.")
        else:
            feedback.pushInfo("📊 Scientific classification report generation was not requested.")

        feedback.setProgress(100)
        feedback.pushInfo("🎯 Enhanced image classification completed successfully! 🎯")
        
        # Summary of results
        feedback.pushInfo("=" * 50)
        feedback.pushInfo("📋 CLASSIFICATION SUMMARY:")
        feedback.pushInfo(f"   🎯 Overall Accuracy: {OA:.4f} ({OA*100:.2f}%)")
        try:
            feedback.pushInfo(f"   🔬 Cohen's Kappa: {kappa:.4f} ({kappa_interp})")
            feedback.pushInfo(f"   ⚖️ Balanced Accuracy: {balanced_acc:.4f}")
        except:
            pass
        feedback.pushInfo(f"   📊 Classes: {len(classes)}")
        feedback.pushInfo(f"   📈 Training samples: {len(feature_list)}")
        feedback.pushInfo(f"   🎛️ Spectral bands: {num_bands}")
        if cv_mean:
            feedback.pushInfo(f"   🔄 Cross-validation: {cv_mean:.4f} ± {cv_std:.4f}")
        feedback.pushInfo("=" * 50)
        
        # Prepare return dictionary
        result_dict = {self.CLASSIFIED_IMAGE: output_file}
        if output_report:
            result_dict[self.OUTPUT_REPORT] = output_report
            
        return result_dict