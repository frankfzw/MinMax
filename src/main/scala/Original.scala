import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils

object SVM {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Original SVM")
    val sc = new SparkContext(conf)
    // Load training data in LIBSVM format.
    val data = MLUtils.loadLibSVMFile(sc, "file:///home/spark/MinMax/data.txt")
    
    // Split data into training (60%) and test (40%).
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)
    
    // Run training algorithm to build the model
    val numIterations = 100
    val positiveTrain = training.filter(r => r.label > 0).randomSplit(Array(0.25, 0.25, 0.25, 0.25), seed = 11L)
    val negativeTrain = training.filter(r => r.label < 1).randomSplit(Array(0.25, 0.25, 0.25, 0.25), seed = 11L)
    println(s"Size pos ${positiveTrain.length}, neg ${negativeTrain.length}")
    
    // val modelArray = new Array[SVMModel](16)
    // for (i <- 0 until 4) {
    //   for (j <- 0 until 4) {
    //     val tmp = positiveTrain(i).union(negativeTrain(j)).cache()
    //     val m = SVMWithSGD.train(tmp, numIterations)
    //     modelArray(i * 4 + j) = m
    //   }
    // }
    val modelArray = positiveTrain.map {
      pos =>
      val res = negativeTrain.map {
        neg =>
        val tmp = pos.union(neg)
        val m = SVMWithSGD.train(tmp, numIterations)
        m
      }
      res
    }

    println(s"Model Array size ${modelArray.length}")
    // val model = SVMWithSGD.train(training, numIterations)

    // Clear the default threshold.
    // model.clearThreshold()

    // Compute raw scores on the test set.
    // val scoreAndLabels = test.map { point =>
    //   val score = model.predict(point.features)
    //     (score, point.label)
    // }

    // Get evaluation metrics.
    // val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    // val auROC = metrics.areaUnderROC()

    // println("Area under ROC = " + auROC)


    // Save and load model
    // model.save(sc, "myModelPath")
    // val sameModel = LogisticRegressionModel.load(sc, "myModelPath")
  }
}


