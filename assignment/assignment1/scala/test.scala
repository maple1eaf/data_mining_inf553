/*
$ sbt package
$ spark-submit --class test ./target/scala-2.11/hw1_2.11-0.1-SNAPSHOT.jar
*/
import java.io.PrintWriter
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization

object test {
    def main(args: Array[String]) {
        // test null
        // val str_1 = """{"business_id":"xOR85RicYj642O3_iJ7hgg","name":"Phoenix Valuations","address":"6340 E Thomas Rd","city":"Scottsdale","state":"AZ","postal_code":"85251","latitude":33.480373,"longitude":-111.9456501,"stars":3.5,"review_count":3,"is_open":1,"attributes":null,"categories":null,"hours":null}"""
        // println(str_1)

        // implicit val formats = DefaultFormats
        // val m1 = parse(str_1).extract[Map[String, Any]]
        // println(m1)
        // println(m1("categories") == null)

        // test task2 split
        // val str_2 = "   wefwefa , fawef,  efaef efawe ,fewafa      "
        // val l_str = str_2.
        
    }
}