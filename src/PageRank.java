import java.io.IOException;
import java.util.*;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.util.*;

public class PageRank {
    private static final int NUM_OF_REDUCER_ITERATIONS = 100;
    private static final double DAMPING_FACTOR = 0.85;
    private static final double TERMINATION_RESIDUAL = 0.001;
    private static final int NUM_NODES = 685230;
    private static final int RESIDUAL_OFFSET = 1000000000;


    public static enum COUNTERS {
        RESIDUAL_SUM
    };

    private static long blockIDofNode(long nodeID) {
        Long n = new Long(nodeID);
        return n.hashCode()%68;
    }

    private static int getBlockNum(int nodeNum) {
        if(nodeNum >= 685230) {
            return 68;
        }

        int[] blockBounds = {10328, 20373, 30629, 40645, 50462, 60841, 70591, 80118, 90497, 100501,
                             110567, 120945, 130999, 140574, 150953, 161332, 171154, 181514, 191625, 202004,
                             212383, 222762, 232593, 242878, 252938, 263149, 273210, 283473, 293255, 303043,
                             313370, 323522, 333883, 343663, 353645, 363929, 374236, 384554, 394929, 404712,
                             414617, 424747, 434707, 444489, 454285, 464398, 474196, 484050, 493968, 503752,
                             514131, 524510, 534709, 545088, 555467, 565846, 576225, 586604, 596585, 606367,
                             616148, 626448, 636240, 646022, 655804, 665666, 675448, 685230
                            };

        int prevCounter = 68;
        int counter = 0;
        Double diff;
        while(true) {

            if(nodeNum == blockBounds[counter]) {
                return counter + 1;
            } else if (nodeNum < blockBounds[counter]) {
                diff = new Double(Math.abs((counter-prevCounter)/2));
                if(diff < 1) {
                    return counter;
                }
                prevCounter = counter;
                counter = counter/2;

            } else {
                diff = new Double(Math.abs((counter-prevCounter)/2));
                if(diff < 1) {
                    return counter + 1;
                }
                prevCounter = counter;
                counter = counter + diff.intValue(); // intValue casts down
            }
        }
    }

    public static class Map extends MapReduceBase implements Mapper<LongWritable, Text, IntWritable, Text> {
        public void map(LongWritable key, Text value, OutputCollector<IntWritable, Text> output, Reporter reporter) throws IOException {
            String[] words = value.toString().split("\\s+");

            int blockNumOfStart = getBlockNum(Integer.valueOf(words[0]));
            int blockNumOfEnd = getBlockNum(Integer.valueOf(words[1]));

            if (blockNumOfStart == blockNumOfEnd) {
                output.collect(new IntWritable(blockNumOfStart), value);
            } else {
                output.collect(new IntWritable(blockNumOfStart), value);
                output.collect(new IntWritable(blockNumOfEnd), value);
            }
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
            Set<Integer> b = new HashSet<Integer>();
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
                    if (be.containsKey(endNode)) {
                        newList = be.get(endNode);
                    } else {
                        newList = new ArrayList<Integer>();
                    }
                    newList.add(startNode);
                    be.put(endNode, newList);
                } else {
                    List<Integer> newList;
                    if (bc.containsKey(endNode)) {
                        newList = bc.get(endNode);
                    } else {
                        newList = new ArrayList<Integer>();
                    }
                    newList.add(startNode);
                    bc.put(endNode, newList);
                }
            }

            java.util.Map<Integer, Double> originalPr = new HashMap<Integer, Double>(pr);

            for (int i = 0; i < NUM_OF_REDUCER_ITERATIONS; i++) {
                java.util.Map<Integer, Double> npr = new HashMap<Integer, Double>();
                for (Integer v : b) {
                    npr.put(v, 0.0);
                }

                for (Integer v : b) {
                    List<Integer> startNodes = be.get(v);
                    if (startNodes != null) {
                        for (Integer u : startNodes) {
                            npr.put(v, npr.get(v) + pr.get(u) / deg.get(u));
                        }
                    }

                    startNodes = bc.get(v);
                    if (startNodes != null) {
                        for (Integer u : startNodes) {
                            npr.put(v, npr.get(v) + pr.get(u) / deg.get(u));
                        }
                    }
                    npr.put(v, npr.get(v) * DAMPING_FACTOR + (1.0 - DAMPING_FACTOR) / NUM_NODES);
                }

                for (Integer v : b) {
                    pr.put(v, npr.get(v));
                }
            }

            for (Integer v : be.keySet()) {
                for (Integer u : be.get(v)) {
                    Double pageRank = pr.get(u);
                    Integer degree = deg.get(u);
                    output.collect(null, new Text(String.format(
                                                      "%s   %s   %s   %s",
                                                      u.toString(), v.toString(),
                                                      pageRank.toString(), degree.toString())));
                }
            }

            Double residualSum = new Double(0);
            for (Integer v : b) {
                residualSum += Math.abs(originalPr.get(v) - pr.get(v)) / pr.get(v);
            }

            reporter.incrCounter(COUNTERS.RESIDUAL_SUM, (int)(residualSum * RESIDUAL_OFFSET));
        }
    }

    public static void main(String[] args) throws Exception {
        double residualSum;
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
            residualSum = ((double)counters.getCounter(COUNTERS.RESIDUAL_SUM))/RESIDUAL_OFFSET;
            System.out.println(String.format("Pass %d: residual = %f", passCount, residualSum));
            passCount += 1;
            if (passCount > 5) break;
        } while (residualSum > (NUM_NODES * TERMINATION_RESIDUAL));

        System.out.println("Total number of MapReduce passes: " + passCount);
    }
}