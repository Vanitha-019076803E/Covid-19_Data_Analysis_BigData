import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, CrossValidatorModel}
import org.apache.spark.ml.param.ParamMap
import java.text.SimpleDateFormat
import java.util.Date
import org.apache.spark.ml.evaluation._
import org.apache.spark.sql.Row
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.{PipelineModel, Pipeline}
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import vegas.sparkExt._
import vegas._
import vegas.render.WindowRenderer._

// Data Preprocessing

// Reading Covid-19 time series data of confirmed cases

val dfConfirmed = spark.read.options(Map("inferSchema"->"true","delimiter"->",","header"->"true")).csv("/home/users/vvaradharajan/Vanitha/BigdataNew/time_series_covid19_confirmed_global.csv")
dfConfirmed.show()
dfConfirmed.count() // 266

// changing dates from columns to rows
def toLongConfirmed(dfConfirmed: DataFrame, by: Seq[String]): DataFrame = {
  val (cols, types) = dfConfirmed.dtypes.filter{ case (c, _) => !by.contains(c)}.unzip
  require(types.distinct.size == 1, s"${types.distinct.toString}.length != 1")

  val expl = explode(array(
    cols.map(c => struct(lit(c).alias("Date"), col(c).alias("Confirmed"))): _*
  ))
  val exprs = by.map(col(_))
  dfConfirmed
    .select(exprs :+ expl.alias("_expl"): _*)
    .select(exprs ++ Seq($"_expl.Date", $"_expl.Confirmed"): _*)
}

val resConfirmed = toLongConfirmed(dfConfirmed, Seq("Province/State","Country/Region","Lat","Long"))
resConfirmed.show()
resConfirmed.count() // 58520

// Reading Covid-19 time series data of death cases
val dfDeaths = spark.read.options(Map("inferSchema"->"true","delimiter"->",","header"->"true")).csv("/home/users/vvaradharajan/Vanitha/BigdataNew/time_series_covid19_deaths_global.csv")
dfDeaths.show()
dfDeaths.count() // 266

// changing dates from columns to rows
def toLongDeath(dfDeaths: DataFrame, by: Seq[String]): DataFrame = {
  val (cols, types) = dfDeaths.dtypes.filter{ case (c, _) => !by.contains(c)}.unzip
  require(types.distinct.size == 1, s"${types.distinct.toString}.length != 1")

  val expl = explode(array(
    cols.map(c => struct(lit(c).alias("Date"), col(c).alias("Death"))): _*
  ))
  val exprs = by.map(col(_))
  dfDeaths
    .select(exprs :+ expl.alias("_expl"): _*)
    .select(exprs ++ Seq($"_expl.Date", $"_expl.Death"): _*)
}

val resDeaths = toLongDeath(dfDeaths, Seq("Province/State","Country/Region","Lat","Long"))
resDeaths.show()
resDeaths.count() // 58520

// Reading Covid-19 time series data of recovered cases
val dfRecovered = spark.read.options(Map("inferSchema"->"true","delimiter"->",","header"->"true")).csv("/home/users/vvaradharajan/Vanitha/BigdataNew/time_series_covid19_recovered_global.csv")
dfRecovered.show()
dfRecovered.count() // 253

// changing dates from columns to rows
def toLongRecovered(dfRecovered: DataFrame, by: Seq[String]): DataFrame = {
  val (cols, types) = dfRecovered.dtypes.filter{ case (c, _) => !by.contains(c)}.unzip
  require(types.distinct.size == 1, s"${types.distinct.toString}.length != 1")

  val expl = explode(array(
    cols.map(c => struct(lit(c).alias("Date"), col(c).alias("Recovered"))): _*
  ))
  val exprs = by.map(col(_))
  dfRecovered
    .select(exprs :+ expl.alias("_expl"): _*)
    .select(exprs ++ Seq($"_expl.Date", $"_expl.Recovered"): _*)
}

val resRecovered = toLongRecovered(dfRecovered, Seq("Province/State","Country/Region","Lat","Long"))
resRecovered.show()
resRecovered.count() // 55660

// dropping state column since more of it's values are null
val resConfirmed1 = resConfirmed.drop("Province/State")
val resDeaths1 = resDeaths.drop("Province/State")
val resRecovered1 = resRecovered.drop("Province/State")

// Renaming column names
val confirmedColRenamed = resConfirmed1.withColumnRenamed("Country/Region","Country").withColumnRenamed("Lat", "Latitude").withColumnRenamed("Long", "Longitude")
val deathsColRenamed = resDeaths1.withColumnRenamed("Country/Region","Country").withColumnRenamed("Lat", "Latitude").withColumnRenamed("Long", "Longitude")
val recoveredColRenamed = resRecovered1.withColumnRenamed("Country/Region","Country").withColumnRenamed("Lat", "Latitude").withColumnRenamed("Long", "Longitude")

val confirmed = confirmedColRenamed.groupBy("Country","Date").sum("Confirmed")
val deaths = deathsColRenamed.groupBy("Country","Date").sum("Death")
val recovered = recoveredColRenamed.groupBy("Country","Date").sum("Recovered")

val confirmedSort = confirmed.sort(col("Country").asc, col("Date").asc)
val deathSort = deaths.sort(col("Country").asc, col("Date").asc)
val recoveredSort = recovered.sort(col("Country").asc, col("Date").asc)

// joining confirmed, death and recovered dataframes into a single dataframe
val consolidatedTimeSeriesData = confirmedSort.join(deathSort, Seq("Country", "Date")).join(recoveredSort, Seq("Country","Date"))
val consolidatedTimeSeriesDataRenamed = consolidatedTimeSeriesData.withColumnRenamed("sum(Confirmed)","Confirmed").withColumnRenamed("sum(Death)", "Death").withColumnRenamed("sum(Recovered)", "Recovered")
consolidatedTimeSeriesDataRenamed.show()

// finding active cases for all the countries
val timeSeriesDataWithActive = consolidatedTimeSeriesDataRenamed.withColumn("Active", col("Confirmed")-col("Death")-col("Recovered"))
timeSeriesDataWithActive.show()

val chinaData = timeSeriesDataWithActive.filter($"Country" === "China")
chinaData.show()

val confirmedDataNullDropped = confirmedColRenamed.na.drop()
val deathDataNullDropped = deathsColRenamed.na.drop()
val recoveredDataNullDropped = recoveredColRenamed.na.drop()

// splitting datas into train data(Datas which have dates <= "6/30/20") and test data(Datas which have dates >= "7/1/20")
val trainRawData = confirmedDataNullDropped.filter(col("Date") <= "6/30/20")
val testRawData = confirmedDataNullDropped.filter(col("Date") >= "7/1/20")

val trainDataTimeStamp = trainRawData.withColumn("dateType_timestamp", to_timestamp(col("Date"), "M/d/yy"))
val testDataTimeStamp = testRawData.withColumn("dateType_timestamp", to_timestamp(col("Date"), "M/d/yy"))

val formattedTrainData = trainDataTimeStamp.withColumn("dateType_timestamp",col("dateType_timestamp").cast("long"))
val oneTrainData = formattedTrainData.filter($"Country" === "Luxembourg") // taking luxembourg datas to do regression
val trainData = oneTrainData.drop("Country", "Date")
val inputCols = trainData.columns.filter(_ != "Confirmed")

val formattedTestData = testDataTimeStamp.withColumn("dateType_timestamp",col("dateType_timestamp").cast("long"))
val oneTestData = formattedTestData.filter($"Country" === "Luxembourg")
val testData = oneTestData.drop("Country", "Date")

trainData.cache()
testData.cache()

// Random Forest Regression

val assembler = new VectorAssembler().setInputCols(inputCols).setOutputCol("featureVector")
val classifier = new RandomForestRegressor().
  setNumTrees(15).
  setFeatureSubsetStrategy("auto").
  setLabelCol("Confirmed").
  setFeaturesCol("featureVector").
  setSeed(10)

val pipeline = new Pipeline().setStages(Array(assembler, classifier))
val paramGrid = new ParamGridBuilder().
  addGrid(classifier.impurity, Array("variance")).
  addGrid(classifier.maxDepth, Array(10, 20)).
  addGrid(classifier.maxBins, Array(10, 100, 300)).build()

val evaluator = new RegressionEvaluator().
  setLabelCol("Confirmed").
  setPredictionCol("prediction").
  setMetricName("rmse")

val cv = new CrossValidator().
  setEstimator(pipeline).
  setEvaluator(evaluator).
  setEstimatorParamMaps(paramGrid).
  setNumFolds(5)

val model = cv.fit(trainData)
val rfModel = model.bestModel.asInstanceOf[PipelineModel].stages.last.asInstanceOf[RandomForestRegressionModel]
println(s"Learned regression forest model:\n ${rfModel.extractParamMap()}")
val t0 = System.nanoTime()
val predictions = model.transform(testData)
val t1 = System.nanoTime()
println("Elapsed time: " + (t1 - t0) + "ns")
predictions.show()

//Linear Regression
val features = new VectorAssembler().setInputCols(Array("dateType_timestamp")).setOutputCol("features")
val lr = new LinearRegression().setLabelCol("Confirmed")
val pipelineLr = new Pipeline().setStages(Array(features, lr))
val lrModel = pipelineLr.fit(trainData)
val linRegModel = lrModel.stages(1).asInstanceOf[LinearRegressionModel]
println(s"RMSE:  ${linRegModel.summary.rootMeanSquaredError}")
println(s"r2:    ${linRegModel.summary.r2}")
println(s"Model: Y = ${linRegModel.coefficients(0)} * X + ${linRegModel.intercept}")
linRegModel.summary.residuals.show()
val t0 = System.nanoTime()
val results = lrModel.transform(testData).select("dateType_timestamp", "Confirmed", "prediction")
val t1 = System.nanoTime()
println("Elapsed time: " + (t1 - t0) + "ns")
results.show()

// Prediction for Future Dates in Linear Regression
val trainDataSample = confirmedDataNullDropped.filter($"Country" === "Luxembourg") // Luxembourg data taken for prediction
val timeStampTrainDataSample = trainDataSample.withColumn("dateType_timestamp", to_timestamp(col("Date"), "M/d/yy"))
timeStampTrainDataSample.printSchema()
val trainDataFormatted = timeStampTrainDataSample.drop("Country", "Date")
val trainDataForFuture = trainDataFormatted.withColumn("dateType_timestamp",col("dateType_timestamp").cast("long"))
val inputCols1 = trainDataForFuture.columns.filter(_ != "Confirmed")

val testDataDF = Seq(
  ("Luxembourg", "49.815","6.1296","9/1/20"), ("Luxembourg","49.815", "6.1296","9/2/20"), ("Luxembourg","49.815", "6.1296","9/3/20"), ("Luxembourg","49.815", "6.1296","9/4/20"),
  ("Luxembourg","49.815", "6.1296","9/5/20"), ("Luxembourg","49.815", "6.1296","9/6/20"), ("Luxembourg","49.815", "6.1296","9/7/20"), ("Luxembourg","49.815", "6.1296","9/8/20"),
  ("Luxembourg","49.815", "6.1296","9/9/20"),("Luxembourg","49.815", "6.1296","9/10/20"),("Luxembourg","49.815", "6.1296","9/11/20")
).toDF("Country", "Latitude","Longitude","Date")

val timeStampTestDataDF = testDataDF.withColumn("dateType_timestamp", to_timestamp(col("Date"), "M/d/yy"))
val testDataFormattedDF = timeStampTestDataDF.drop("Country", "Date")
val testDatasDF = testDataFormattedDF.withColumn("dateType_timestamp",col("dateType_timestamp").cast("long")).withColumn("Latitude",col("Latitude").cast("double")).withColumn("Longitude",col("Longitude").cast("double"))

val features1 = new VectorAssembler().setInputCols(Array("dateType_timestamp")).setOutputCol("features")
val lr1 = new LinearRegression().setLabelCol("Confirmed")
val pipelineLr1 = new Pipeline().setStages(Array(features1, lr1))
val lrModel1 = pipelineLr1.fit(trainDataForFuture)
val linRegModel1 = lrModel1.stages(1).asInstanceOf[LinearRegressionModel]
println(s"RMSE:  ${linRegModel1.summary.rootMeanSquaredError}")
println(s"r2:    ${linRegModel1.summary.r2}")
println(s"Model: Y = ${linRegModel1.coefficients(0)} * X + ${linRegModel1.intercept}")
linRegModel1.summary.residuals.show()
val t0 = System.nanoTime()
val resultsForFutureDates = lrModel1.transform(testDatasDF).select("dateType_timestamp", "Latitude","Longitude", "prediction")
val t1 = System.nanoTime()
println("Elapsed time: " + (t1 - t0) + "ns")
val finalResult = resultsForFutureDates.withColumn("dateType_timestamp",col("dateType_timestamp").cast("timestamp"))
finalResult.show()

// showing luxembourg last few days cases to prove that tha prediction is accurate
val luxCurrentCases = resConfirmed.filter($"Country/Region" === "Luxembourg" && $"Date" >= "8/28/20")
luxCurrentCases.show()

//plotting for Linear Regression
val results1 = results.withColumn("dateType_timestamp", col("dateType_timestamp").cast("timestamp"))
val results2 = results1.filter(col("dateType_timestamp") > "2020-07-31 00:00:00")
val plotLinearRegression = Vegas("trend of Country wise Confirmed cases from max to min ",width = 500.0, height = 400.0).
  withDataFrame(results2.select("dateType_timestamp", "prediction")).
  encodeX("dateType_timestamp", Nominal).
  encodeY("prediction",Quantitative).
  mark(Point).encodeColor(field = "symbol", dataType = Nominal,legend = Legend(orient = "left", title = "Linear Regression")).show

//---------------------------Plotting--------------------------

// Trend of Worldwide Confirmed Cases
val plotConfirm = Vegas("world Confirmed Rate",width = 500.0, height = 500.0).
  withDataFrame(confirmedColRenamed.select("Date", "Confirmed")).
  encodeX("Date", Temp).
  encodeY("Confirmed", Quantitative, aggregate = AggOps.Sum, axis = Axis(title = "Confirmed")).
  mark(Line).encodeColor(field = "symbol", dataType = Nominal,legend = Legend(orient = "left", title = "Worldwide Confirmed Rate")).show

// Trend of Worldwide Death Cases
val plotDeath = Vegas("world Death Rate",width = 500.0, height = 500.0).
  withDataFrame(deathsColRenamed.select("Date", "Death")).
  encodeX("Date", Temp).
  encodeY("Death", Quantitative, aggregate = AggOps.Sum, axis = Axis(title = "Death")).
  mark(Line).encodeColor(field = "symbol", dataType = Nominal,legend = Legend(orient = "left", title = "Worldwide Death Rate")).show

// Trend of Worldwide Recovered Cases
val plotRecovered = Vegas("world Recovered Rate",width = 500.0, height = 500.0).
  withDataFrame(recoveredColRenamed.select("Date", "Recovered")).
  encodeX("Date", Temp).
  encodeY("Recovered", Quantitative, aggregate = AggOps.Sum, axis = Axis(title = "Recovered")).
  mark(Line).encodeColor(field = "symbol", dataType = Nominal,legend = Legend(orient = "left", title = "Worldwide Recovered Rate")).show

// Trend of Cases in Luxembourg on daily basis
val luxData = confirmedColRenamed.filter($"Country" === "Luxembourg")
luxData.show()

val plotLux = Vegas("world Confirmed Rate",width = 500.0, height = 500.0).
  withDataFrame(luxData.select("Date", "Confirmed")).
  encodeX("Date", Temp).
  encodeY("Confirmed", Quantitative, aggregate = AggOps.Sum, axis = Axis(title = "Confirmed")).
  mark(Line).encodeColor(field = "symbol", dataType = Nominal,legend = Legend(orient = "left", title = "Luxembourg Confirmed Rate")).show

// Countrywise Confirmed Rate

val currentStatus = timeSeriesDataWithActive.groupBy("Country").max().orderBy(desc("max(Confirmed)"))
currentStatus.show(10)
val max = currentStatus.sort($"max(Confirmed)".desc)
val plot = Vegas("trend of Country wise Confirmed cases from max to min ",width = 400.0, height = 400.0).
  withDataFrame(max.limit(10).select("Country", "max(Confirmed)")).
  encodeX("Country", Ordinal).
  encodeY("max(Confirmed)",Quantitative).
  mark(Bar).encodeColor(field = "symbol", dataType = Nominal,legend = Legend(orient = "center", title = "Country-wise Confirmed Rate (max to min)")).show

// State wise report of confirmed cases in China
val ChinaData = resConfirmed.filter($"Country/Region" === "China")
val chinaConfirmedData = ChinaData.groupBy("Province/State").sum("Confirmed")
val maxChinaConfirmedData = chinaConfirmedData.sort($"sum(Confirmed)".desc)
maxChinaConfirmedData.show(5)

val plotChina = Vegas("China state wise Confirmed cases",width = 1300.0, height = 500.0).
  withDataFrame(maxChinaConfirmedData.limit(10).select("Province/State", "sum(Confirmed)")).
  encodeX("sum(Confirmed)", Quantitative,aggregate = AggOps.Sum, axis = Axis(title = "Confirmed Cases")).
  encodeY("Province/State",Ordinal).
  mark(Bar).encodeColor(field = "symbol", dataType = Nominal,legend = Legend(orient = "left", title = "China(state-wise) Confirmed Rate")).show

// Worldwide confirmed, death, recovered and active cases till now
val timeSeriesDataWithActive1 = timeSeriesDataWithActive.filter($"Date" === "8/28/20")
val totalCasesWorldwide = timeSeriesDataWithActive1.agg(sum("Confirmed"),sum("Death"),sum("recovered"),sum("active"))
val totalCasesWorldwide1 = totalCasesWorldwide.withColumn("sum(Confirmed)", col("sum(Confirmed)").cast("string")).withColumn("sum(Death)", col("sum(Death)").cast("string")).withColumn("sum(Recovered)", col("sum(Recovered)").cast("string")).withColumn("sum(Active)", col("sum(Active)").cast("string"))
def toLongTotal(totalCasesWorldwide1: DataFrame, by: Seq[String]): DataFrame = {
  val (cols, types) = totalCasesWorldwide1.dtypes.filter{ case (c, _) => !by.contains(c)}.unzip
  require(types.distinct.size == 1, s"${types.distinct.toString}.length != 1")

  val expl = explode(array(
    cols.map(c => struct(lit(c).alias("CasesCategory"), col(c).alias("Cases"))): _*
  ))
  val exprs = by.map(col(_))
  totalCasesWorldwide1
    .select(exprs :+ expl.alias("_expl"): _*)
    .select(exprs ++ Seq($"_expl.CasesCategory", $"_expl.Cases"): _*)
}
val totalCasesWorldWide = toLongTotal(totalCasesWorldwide1, Seq())
val totalCasesWorldWide1 = totalCasesWorldWide.withColumn("Cases", col("Cases").cast("long"))
val plotTotalCases = Vegas("Worldwide cases",width = 400.0, height = 400.0).
  withDataFrame(totalCasesWorldWide1.select("CasesCategory", "Cases")).
  encodeX("CasesCategory", Ordinal).
  encodeY("Cases",Quantitative).
  mark(Bar).encodeColor(field = "symbol", dataType = Nominal,legend = Legend(orient = "left", title = "Worldwide cases")).show





