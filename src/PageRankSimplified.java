import java.io.IOException;
import java.util.*;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.util.*;

public class PageRankSimplified {
    private static final int NUM_OF_REDUCER_ITERATIONS = 100;
    private static final double DAMPING_FACTOR = 0.85;
    private static final double TERMINATION_RESIDUAL = 0.001;
    private static final int NUM_NODES = 685230;
    private static final int RESIDUAL_OFFSET = 1000000000;


    public static enum COUNTERS {
        PROCESSED_NODES,
        RESIDUAL_SUM
    };

    private static long blockIDofNode(long nodeID) {
        Long n = new Long(nodeID);
        return n.hashCode()%68;
    }

    private static final int[] BLOCK_BOUNDS = {10328, 20373, 30629, 40645, 50462, 60841, 70591, 80118, 90497, 100501,
                               110567, 120945, 130999, 140574, 150953, 161332, 171154, 181514, 191625, 202004,
                               212383, 222762, 232593, 242878, 252938, 263149, 273210, 283473, 293255, 303043,
                               313370, 323522, 333883, 343663, 353645, 363929, 374236, 384554, 394929, 404712,
                               414617, 424747, 434707, 444489, 454285, 464398, 474196, 484050, 493968, 503752,
                               514131, 524510, 534709, 545088, 555467, 565846, 576225, 586604, 596585, 606367,
                               616148, 626448, 636240, 646022, 655804, 665666, 675448, 685230
                                              };

    private static int getBlockNum(int nodeNum) {
        int blockNum = nodeNum/10000;

        if(nodeNum < 10328) {
            return 0;
        } else {
            if(nodeNum < BLOCK_BOUNDS[blockNum-1]) {
                return (blockNum-1);
            } else {
                return blockNum;
            }
        }
    }

    public static class Map extends MapReduceBase implements Mapper<LongWritable, Text, IntWritable, Text> {
        public void map(LongWritable key, Text value, OutputCollector<IntWritable, Text> output, Reporter reporter) throws IOException {
            String[] words = value.toString().split("\\s+");

            if (words.length == 4) {
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
            int blockNum = key.get();
            Set<Integer> b = new HashSet<Integer>();
            ArrayList<Edge> edges = new ArrayList<Edge>();
            java.util.Map<Integer, Double> pr = new HashMap<Integer, Double>();
            java.util.Map<Integer, Integer> deg = new HashMap<Integer, Integer>();
            java.util.Map<Integer, List<Integer>> be = new HashMap<Integer, List<Integer>>();
            java.util.Map<Integer, List<Integer>> bc = new HashMap<Integer, List<Integer>>();

            while (values.hasNext()) {
                String line = values.next().toString();
                String[] words = line.split("\\s+");
                try {
                    Integer startNode = Integer.valueOf(words[0]);
                    Integer endNode = Integer.valueOf(words[1]);
                    Double pageRank = Double.valueOf(words[2]);
                    Integer degree = Integer.valueOf(words[3]);
                    pr.put(startNode, pageRank);
                    deg.put(startNode, degree);
                    edges.add(new Edge(startNode, endNode));

                    if (getBlockNum(startNode) == blockNum) {
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
                } catch (java.lang.NumberFormatException e) {
                    // Just ignore the exception; it's expected.
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

            for (Integer v : b) {
                reporter.incrCounter(COUNTERS.PROCESSED_NODES, 1);
                reporter.incrCounter(COUNTERS.RESIDUAL_SUM, (int)((Math.abs(originalPr.get(v) - pr.get(v)) / pr.get(v)) * RESIDUAL_OFFSET));
            }

            int highestNode = BLOCK_BOUNDS[key.get()] - 1;
            Double pageRank = pr.get(highestNode);
            if (pageRank == null) {
                output.collect(new IntWritable(1), new Text(String.format("Pagerank of node %d of block %d: null", highestNode, blockNum)));
            } else {
                System.out.println(String.format("Pagerank of node %d of block %d: %.20f", highestNode, blockNum, pageRank));
                output.collect(new IntWritable(1), new Text(String.format("Pagerank of node %d of block %d: %.20f", highestNode, blockNum, pageRank)));
            }
        }
    }

    public static void main(String[] args) throws Exception {
        double residualSum;
        int passCount = 0;
        System.out.println("Starting...");
        do {
            JobConf conf = new JobConf(PageRankSimplified.class);
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
            System.out.println(String.format("Number of nodes processed: %d", counters.getCounter(COUNTERS.PROCESSED_NODES)));
            passCount += 1;
        } while (residualSum > (NUM_NODES * TERMINATION_RESIDUAL));

        System.out.println("Total number of MapReduce passes: " + passCount);
    }
}