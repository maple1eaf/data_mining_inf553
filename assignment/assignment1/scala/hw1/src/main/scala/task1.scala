/*
Scala: $ spark-submit --class task1 hw1.jar <input_file> <output_file> <stopwords> <y> <m> <n>

$ sbt package
$ spark-submit --class task1 ./target/scala-2.11/hw1_2.11-0.1-SNAPSHOT.jar "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw1/hw1_data/review.json" "output_task1_scala.json" "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw1/hw1_data/stopwords" 2018 10 10
*/

import java.io.PrintWriter
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization

object task1 {
    def main(args: Array[String]) {
        // parameters
        val input_file = args(0)
        val output_file =args(1)
        val stopwords_path = args(2)
        val y = args(3)
        val m = args(4).toInt
        val n = args(5).toInt

        // val input_file =  "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw1/hw1_data/review.json"
        // val stopwords_path = "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw1/hw1_data/stopwords"

        val conf = new SparkConf().setAppName("task1").setMaster("local[*]")
        val sc = new SparkContext(conf)

        def textToMap(r: String): Map[String, Any] = {
            implicit val formats = DefaultFormats
            parse(r).extract[Map[String, Any]]
        }
        var review_data = sc.textFile(input_file).map(textToMap).cache()
        // println(review_data)

        // A
        val total_number_of_reviews = review_data.count()
        println(total_number_of_reviews)

        // B
        // val y = "2018"
        val number_of_reviews_in_a_given_year = review_data.filter(r => r("date").toString().startsWith(y)).count()
        println(number_of_reviews_in_a_given_year)

        // C
        val number_of_distinct_users = review_data.map(r => r("user_id")).distinct().count()
        println(number_of_distinct_users)

        // D
        // val m = "5".toInt
        val top_m_users = review_data.map(r => (r("user_id"),1))
                                    .reduceByKey(_+_)
                                    .sortBy(_._2, ascending=false)
                                    .map(x => List(x._1.toString, x._2))
                                    .take(m)
        top_m_users.foreach(println)

        // E
        // val n = "15".toInt
        val PUNCTUATIONS_PATERN = "[\\[\\](),.!?:;\\s]"

        val stopwords = sc.textFile(stopwords_path).collect
        // stopwords.foreach(println)
        val stopwords_with_null_string = ""::stopwords.toList

        val top_n_frequent_words_with_count = review_data.map(r => r("text").toString.toLowerCase())
                                            .map(l => l.split(PUNCTUATIONS_PATERN))
                                            .map(l => l.filter(w => !stopwords_with_null_string.contains(w)))
                                            .flatMap(l => l.map(x => (x, 1)))
                                            .reduceByKey(_+_)
                                            .sortBy(_._1, ascending=true)
                                            .sortBy(_._2, ascending=false)
                                            .take(n)
        val top_n_frequent_words_without_count = top_n_frequent_words_with_count.map(_._1)
        top_n_frequent_words_with_count.foreach(println)

        // result
        val result = Map("A"->total_number_of_reviews, "B"->total_number_of_reviews, "C"->number_of_distinct_users, "D"->top_m_users, "E"->top_n_frequent_words_without_count)
        implicit val formats = DefaultFormats
        val result_json = Serialization.write(result)
        println(result_json)

        // write to output_task1_scala.json file
        // val output = "output_task1_scala.json"
        val out = new PrintWriter(output_file)
        out.print(result_json)
        out.close()


        // val s1 = "Came here on a Thursday night at 6:30 p.m. My friends and I had a reservation, but it was not needed - the place was almost empty.\n\nSERVICE: Extremely Poor\n\nWhile I enjoyed the company of my friends, I would not come back to this restaurant, primarily because of the poor service. Once we were all seated, we were not greeted by anyone for over half an hour, and when someone did come by, we asked whether we could place our orders, to which he said he could not; he's doing something else - so we waited at least another 15 minutes before someone came to finally take our orders. \n\nMAIN: Caprese Salad\n\nThe salad was okay - a bit salty for my liking. I personally did not find this dish filling, so I ordered the sweet potato fries as well (See below).\n\nMAIN: Sweet Potato Fries (this was worth it) \n\nThese fries were actually very delicious - crispy on the outside and soft on the inside. If you visit this restaurant, I would recommend this particular dish, especially given the price ($8) and the portion size (it was quite large - almost the size of a large plate)."
        // val s2 = "Very enjoyable experience here.  During the week till 8 pm they have a 'bar menu' with lower priced items, there's like 7 sushi rolls priced at $10, specialty cocktails-including HH sake #score, ramen noodles and some appies. \n\nI got the angry tuna roll and it was divine. It is on the smaller side so you will have to order something else to satiate your appetitie, but I was OK with not choking on the size of the sushi when trying to consume it!  I got the ramen soup next, which is like my new favorite thing, and it was really tasty as well. \n\nThe bartender that was serving us was excellent, gave us his recommendations [including what not to order] and was extremely attentive.  Other great news about sitting at the bar is they have TV's for football watching!!! Hooray TV's and Football! \n\nMy friend got a roll and the Kobe sliders which he must've really enjoyed b/c he housed them both!\n\nDef on my list of I would go again!"
        // val l1 = s2.split(PUNCTUATIONS_PATERN)
        // l1.foreach(println)
    }
}