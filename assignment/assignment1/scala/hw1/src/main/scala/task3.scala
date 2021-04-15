/*
Scala: $ spark-submit --class task3 hw1.jar <input_file> <output_file> <partition_type> <n_partitions> <n>

$ sbt package
// default partitioner
$ spark-submit --class task3 ./target/scala-2.11/hw1_2.11-0.1-SNAPSHOT.jar "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw1/hw1_data/review.json" "output_task3_scala.json" "default" 30 150
// customized partitioner
$ spark-submit --class task3 ./target/scala-2.11/hw1_2.11-0.1-SNAPSHOT.jar "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw1/hw1_data/review.json" "output_task3_scala.json" "customized" 30 150
*/

import java.io.PrintWriter
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.Partitioner

import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization

class BusinessidPartitioner(numParts:Int) extends Partitioner{
    override def numPartitions: Int = numParts

    override def getPartition(key: Any): Int = {
        key.toString.hashCode % numPartitions
    }

    override def equals(other: Any): Boolean = other match {
		case t: BusinessidPartitioner => t.numPartitions == numPartitions
		case _ => false
	}
}

object task3 {
    def main(args: Array[String]) {
        // parameters
        val input_file = args(0)
        val output_file = args(1)
        val partition_type = args(2)
        val n_partitions = args(3).toInt
        val n = args(4).toInt

        // val input_file =  "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw1/hw1_data/review.json"
        // val stopwords_path = "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw1/hw1_data/stopwords"

        val conf = new SparkConf().setAppName("task3").setMaster("local[*]")
        val sc = new SparkContext(conf)

        def textToMapBusinessId(r: String): (String, Int) = {
            implicit val formats = DefaultFormats
            val m = parse(r).extract[Map[String, Any]]
            (m("business_id").toString, 1)
        }


        var business_id_data = sc.textFile(input_file).map(textToMapBusinessId)

        if (partition_type == "default") {
            business_id_data = business_id_data.cache()
        } else if (partition_type == "customized") {
            business_id_data = business_id_data.partitionBy(new BusinessidPartitioner(n_partitions)).cache()
        } else {
            printf("Wrong partition type.")
        }
        
        // println(business_id_data.getClass)

        // number_of_partitions
        val number_of_partitions = business_id_data.getNumPartitions
        println(number_of_partitions)

        // number_of_items_in_each_partition
        def countItemsInAPartition(iter: Iterator[Any]): Iterator[Int] = {
            val count = List[Int]()
            val i = iter.length
            count.::(i).iterator
        }
        val number_of_items_in_each_partition = business_id_data.mapPartitions(countItemsInAPartition).collect
        number_of_items_in_each_partition.foreach(println)

        // businesses_that_have_more_than_n_reviews
        val businesses_that_have_more_than_n_reviews = business_id_data.reduceByKey(_+_)
                                                                    .filter(_._2 > n)
                                                                    .map(x => List(x._1.toString, x._2))
                                                                    .collect
        businesses_that_have_more_than_n_reviews.foreach(println)

        // result
        val result = Map("n_partitions"->number_of_partitions, "n_items"->number_of_items_in_each_partition, "result"->businesses_that_have_more_than_n_reviews)
        implicit val formats = DefaultFormats
        val result_json = Serialization.write(result)
        println(result_json)

        // write to output_task3_scala.json file
        val out = new PrintWriter(output_file)
        out.print(result_json)
        out.close()

    }
}