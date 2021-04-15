/*
Scala: $ spark-submit --class task2 hw1.jar <review_file> <business_file> <output_file> <if_spark> <n>

$ sbt package
// spark
$ spark-submit --class task2 ./target/scala-2.11/hw1_2.11-0.1-SNAPSHOT.jar "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw1/hw1_data/review.json" "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw1/hw1_data/business.json" "output_task2_spark_scala.json" "spark" 30
// no_spark
$ spark-submit --class task2 ./target/scala-2.11/hw1_2.11-0.1-SNAPSHOT.jar "/Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw1/hw1_data/review.json" "/Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw1/hw1_data/business.json" "output_task2_nospark_scala.json" "no_spark" 30
*/

import scala.io.Source
import java.io.PrintWriter
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization



object task2 {
    def main(args: Array[String]) {
        // parameters
        val review_file = args(0)
        val business_file =args(1)
        val output_file = args(2)
        val if_spark = args(3)
        val n = args(4).toInt

        def withSpark() {
            val conf = new SparkConf().setAppName("task2").setMaster("local[*]")
            val sc = new SparkContext(conf)

            def textToMap(r: String): Map[String, Any] = {
                implicit val formats = DefaultFormats
                parse(r).extract[Map[String, Any]]
            }

            def businessWithOneCategory(t: (String, String)): List[(String, String)] = {
                val business: String = t._1
                val category: String = t._2
                val cate_array: Array[String] = category.split(",").map(_.trim())
                var bc = List[(String, String)]()
                for (c <- cate_array) {
                    bc = bc:::List((business, c))
                }
                bc
            }
            val business_data = sc.textFile(business_file)
                                .map(textToMap)
                                .filter(b => b("categories") != null)
                                .flatMap(b => businessWithOneCategory((b("business_id").toString, b("categories").toString)))
            // business_data.foreach(println)

            val review_data = sc.textFile(review_file)
                                .map(textToMap)
                                .map(r => (r("business_id").toString, (r("stars").asInstanceOf[Number].floatValue, 1)))
                                .reduceByKey((x, y) => ((x._1+y._1), (x._2+y._2)))
            // review_data.foreach(println)

            val data_join_on_bussiness_id = business_data.join(review_data)
                                                        .map(_._2)
                                                        .reduceByKey((x, y) => ((x._1+y._1), (x._2+y._2)))
                                                        .mapValues(t => t._1/t._2)
                                                        .sortBy(_._1, ascending=true)
                                                        .sortBy(_._2, ascending=false)
                                                        .map(x => List(x._1, x._2))
                                                        .take(n)
            data_join_on_bussiness_id.foreach(println)

            // result
            val result = Map("result" -> data_join_on_bussiness_id)
            implicit val formats = DefaultFormats
            val result_json = Serialization.write(result)
            println(result_json)
            
            // write to output_task2_scala.json file
            // val output = "output_task2_scala.json"
            val out = new PrintWriter(output_file)
            out.print(result_json)
            out.close()
        }

        def withoutSpark() {
            // process review.json
            val review_file_pointer = Source.fromFile(review_file)
            val review_file_lines = review_file_pointer.getLines

            def processReviewLines(r: String): (String, (Float, Int)) = {
                implicit val formats = DefaultFormats
                val rm = parse(r).extract[Map[String, Any]]
                (rm("business_id").toString, (rm("stars").asInstanceOf[Number].floatValue, 1))
            }

            def sumAndCountValue(x:(String, List[(String, (Float, Int))])): (String, (Float, Int))= {
                val key = x._1
                val l = x._2
                var sum: Float = 0.0f
                var count: Int = 0
                for (item <- l) {
                    sum += item._2._1
                    count += item._2._2
                }
                (key, (sum, count))
            }
            val review_data_map = review_file_lines.map(processReviewLines)
                                            .toList
                                            .groupBy(_._1)
                                            .map(x => sumAndCountValue(x))
                                            .toMap
            // println(review_data_map)
            
            // process business.json
            val business_file_pointer = Source.fromFile(business_file)
            val business_file_lines = business_file_pointer.getLines

            def parseBusinessLines(b: String): Map[String, Any] = {
                implicit val formats = DefaultFormats
                parse(b).extract[Map[String, Any]]
            }

            def processBusinessData(b: Map[String, Any]): List[(String, String)] = {
                val business = b("business_id").toString
                val categories = b("categories").toString
                val cate_array = categories.split(",").map(_.trim())
                var bc = List[(String, String)]()
                for (c <- cate_array) {
                    bc = bc:::List((c, business))
                }
                bc
            }
            val business_data = business_file_lines.map(parseBusinessLines)
                                                .filter(b => b("categories") != null)
                                                .flatMap(processBusinessData)
                                                .filter(x => review_data_map.get(x._2) != None)
            
            // join business and review on business_id
            def getSumCountFromReview(b: (String, String)): (String, (Float, Int)) = {
                val category = b._1
                val business = b._2
                val sum_count = review_data_map(business)
                (category, sum_count)
            }

            val result_after_join = business_data.map(getSumCountFromReview)
                                                .toList
                                                .groupBy(_._1)
                                                .map(sumAndCountValue)
                                                .map(x => (x._1, x._2._1/x._2._2))
                                                .toList
                                                .sortBy(_._1).reverse
                                                .sortBy(_._2).reverse
                                                .map(x => List(x._1, x._2))
                                                .take(n)

            // result
            val result = Map("result" -> result_after_join)
            implicit val formats = DefaultFormats
            val result_json = Serialization.write(result)
            println(result_json)
            
            // write to output file
            val out = new PrintWriter(output_file)
            out.print(result_json)
            out.close()
        }
        
        if (if_spark == "spark"){
            withSpark()
        } else if (if_spark == "no_spark") {
            withoutSpark()
        } else {
            println("Wrong if_spark value.")
        }
    }
}