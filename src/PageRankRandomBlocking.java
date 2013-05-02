import java.io.IOException;
import java.util.*;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.util.*;

public class PageRankRandomBlocking {
    private static final int NUM_OF_REDUCER_ITERATIONS = 500;
    private static final double DAMPING_FACTOR = 0.85;
    private static final double TERMINATION_RESIDUAL = 0.001;
    private static final int NUM_NODES = 685230;
    private static final int RESIDUAL_OFFSET = 1000000000;

    public static enum COUNTERS {
        PROCESSED_NODES,
        RESIDUAL_SUM
    };

    private static int getBlockNum(int nodeID) {
        if (nodeID < 0) return -1;
        Long n = new Long(nodeID);
        return n.hashCode()%68;
    }

    public static class Map extends MapReduceBase implements Mapper<LongWritable, Text, IntWritable, Text> {
        public void map(LongWritable key, Text value, OutputCollector<IntWritable, Text> output, Reporter reporter) throws IOException {
            String[] words = value.toString().split("\\s+");

            int blockNumOfStart = getBlockNum(Integer.valueOf(words[0]));
            int blockNumOfEnd = getBlockNum(Integer.valueOf(words[1]));

            if (blockNumOfEnd == -1) {
                output.collect(new IntWritable(blockNumOfStart), value);
            } else if (blockNumOfStart == blockNumOfEnd) {
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
        }
    }

    public static void main(String[] args) throws Exception {
        double residualSum;
        int passCount = 0;
        System.out.println("Starting...");
        do {
            JobConf conf = new JobConf(PageRankRandomBlocking.class);
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