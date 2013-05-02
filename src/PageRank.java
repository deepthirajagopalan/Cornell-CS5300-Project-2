package cs5300;

import java.io.IOException;
import java.util.*;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.util.*;

public class PageRank {
    private static final int NUM_OF_REDUCER_ITERATIONS = 10;
    private static final double DAMPING_FACTOR = 0.85;
    private static final double TERMINATION_RESIDUAL = 0.001;
    private static final int NUM_NODES = 685229;
    private static final int RESIDUAL_OFFSET = 1000000000;


    public static enum COUNTERS {

		RESIDUAL_SUM
	};
	
	private static long blockIDofNode(long nodeID) { 
		Long n = new Long(nodeID);
		return n.hashCode()%68; 
	} 


    // private static int getBlockNum(int nodeNum) {
    //     if(nodeNum >= 685230) {
    //         return 68;
    //     }

    //     int[] blockBounds = {10328, 20373, 30629, 40645, 50462, 60841, 70591, 80118, 90497, 100501,
    //                          110567, 120945, 130999, 140574, 150953, 161332, 171154, 181514, 191625, 202004,
    //                          212383, 222762, 232593, 242878, 252938, 263149, 273210, 283473, 293255, 303043,
    //                          313370, 323522, 333883, 343663, 353645, 363929, 374236, 384554, 394929, 404712,
    //                          414617, 424747, 434707, 444489, 454285, 464398, 474196, 484050, 493968, 503752,
    //                          514131, 524510, 534709, 545088, 555467, 565846, 576225, 586604, 596585, 606367,
    //                          616148, 626448, 636240, 646022, 655804, 665666, 675448, 685230
    //                         };

    //     int prevCounter = 68;
    //     int counter = 0;
    //     Double diff;
    //     while(true) {

    //         if(nodeNum == blockBounds[counter]) {
    //             return counter + 1;
    //         } else if (nodeNum < blockBounds[counter]) {
    //             diff = new Double(Math.abs((counter-prevCounter)/2));
    //             if(diff < 1) {
    //                 return counter;
    //             }
    //             prevCounter = counter;
    //             counter = counter/2;

    //         } else {
    //             diff = new Double(Math.abs((counter-prevCounter)/2));
    //             if(diff < 1) {
    //                 return counter + 1;
    //             }
    //             prevCounter = counter;
    //             counter = counter + diff.intValue();	// intValue casts down
    //         }


    //     }
    // }


    private static int getBlockNum(int nodeNum) {
        if (nodeNum < 2) {
            return 0;
        }
        else if (nodeNum < 4) {
            return 1;
        }
        else if (nodeNum < 6) {
            return 2;
        }
        else {
            return 3;
        }
    }

    public static class Map extends MapReduceBase implements Mapper<LongWritable, Text, IntWritable, Text> {
        public void map(LongWritable key, Text value, OutputCollector<IntWritable, Text> output, Reporter reporter) throws IOException {
            String[] words = value.toString().split("\\s+");

            output.collect(new IntWritable(getBlockNum(Integer.valueOf(words[0]))), value);
            output.collect(new IntWritable(getBlockNum(Integer.valueOf(words[1]))), value);
        }
    }

    private static class Edge {
        public Integer startNode;
        public Integer endNode;

        public Edge(Integer startNode, Integer endNode) {
            this.startNode = startNode;
            this.endNode = endNode;
        }
    }

    public static class Reduce extends MapReduceBase implements Reducer<IntWritable, Text,  IntWritable, Text> {
        public void reduce(IntWritable key, Iterator<Text> values, OutputCollector<IntWritable, Text> output, Reporter reporter) throws IOException {
            ArrayList<Integer> b = new ArrayList<Integer>();
            ArrayList<Edge> edges = new ArrayList<Edge>();
            java.util.Map<Integer, Double> pr = new HashMap<Integer, Double>();
            java.util.Map<Integer, Integer> deg = new HashMap<Integer, Integer>();
            java.util.Map<Integer, List<Integer>> be = new HashMap<Integer, List<Integer>>();
            java.util.Map<Integer, List<Integer>> bc = new HashMap<Integer, List<Integer>>();

            while (values.hasNext()) {
                String line = values.next().toString();
                String[] words = line.split("\\s+");
                Integer startNode = Integer.valueOf(words[0]);
                Integer endNode = Integer.valueOf(words[1]);
                Double pageRank = Double.valueOf(words[2]);
                Integer degree = Integer.valueOf(words[3]);
                pr.put(startNode, pageRank);
                deg.put(startNode, degree);
                edges.add(new Edge(startNode, endNode));

                if (getBlockNum(startNode) == key.get()) {
                    b.add(startNode);
                    List<Integer> newList;
                    if (be.containsKey(startNode)) {
                        newList = be.get(startNode);
                    } else {
                        newList = new ArrayList<Integer>();
                    }
                    newList.add(endNode);
                    be.put(startNode, newList);
                } else {
                    List<Integer> newList;
                    if (bc.containsKey(startNode)) {
                        newList = bc.get(startNode);
                    } else {
                        newList = new ArrayList<Integer>();
                    }
                    newList.add(endNode);
                    bc.put(startNode, newList);
                }
            }

            java.util.Map<Integer, Double> originalPr = new HashMap<Integer, Double>(pr);

            for (int i = 0; i < NUM_OF_REDUCER_ITERATIONS; i++) {
                java.util.Map<Integer, Double> npr = new HashMap<Integer, Double>();
                for (int j = 0; j < b.size(); j++) {
                    npr.put(b.get(j), 0.0);
                }
                for (int p = 0; p < edges.size(); p++) {
                    Edge edge = edges.get(p);
                    Integer startNode = edge.startNode;
                    Integer endNode = edge.endNode;
                    npr.put(endNode, npr.get(endNode) + pr.get(startNode) / deg.get(startNode));
                }
                for (int j = 0; j < b.size(); j++) {
                    Integer v = b.get(j);
                    npr.put(v, npr.get(v) * DAMPING_FACTOR + (1 - DAMPING_FACTOR) / NUM_NODES);
                    pr.put(v, npr.get(v));
                }
            }

            for (int i = 0; i < b.size(); i++) {
                Integer v = b.get(i);
                List<Integer> endNodes = be.get(v);
                Double pageRank = pr.get(v);
                Integer degree = deg.get(v);
                for (int j = 0; j < endNodes.size(); j++) {
                    Integer endNode = endNodes.get(j);
                    output.collect(null, new Text(String.format(
                                                      "%s   %s   %s   %s",
                                                      v.toString(), endNode.toString(),
                                                      pageRank.toString(), degree.toString())));
                }
            }

            Double residualSum = new Double(0);
            for (int i = 0; i < b.size(); i++) {
                Integer v = b.get(i);
                residualSum += Math.abs(originalPr.get(i) - pr.get(i)) / pr.get(i);
            }

            reporter.incrCounter(COUNTERS.RESIDUAL_SUM, (int)(residualSum * RESIDUAL_OFFSET));
        }
    }

    public static void main(String[] args) throws Exception {
    	// 
        float residualSum;
        int passCount = 0;
        do {
            JobConf conf = new JobConf(PageRank.class);
            conf.setJobName("pagerank");

            conf.setOutputKeyClass(IntWritable.class);
            conf.setOutputValueClass(Text.class);

            conf.setMapperClass(Map.class);
            conf.setReducerClass(Reduce.class);

            conf.setInputFormat(TextInputFormat.class);
            conf.setOutputFormat(TextOutputFormat.class);

            FileInputFormat.setInputPaths(conf, new Path(args[0] + passCount));
            FileOutputFormat.setOutputPath(conf, new Path(args[0] + (passCount + 1)));

            RunningJob job = JobClient.runJob(conf);
            Counters counters = job.getCounters();
            residualSum = ((float)counters.getCounter(COUNTERS.RESIDUAL_SUM))/RESIDUAL_OFFSET;
            System.out.println(String.format("Pass %d: residual = %f", passCount, residualSum));
            passCount += 1;
        } while (residualSum < 0.001);

        System.out.println("Total number of MapReduce passes: " + passCount);
    }
}