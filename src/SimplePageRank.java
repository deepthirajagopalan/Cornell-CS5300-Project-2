import java.io.IOException;
import java.util.*;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.util.*;

import com.sun.xml.internal.bind.CycleRecoverable.Context;

public class SimplePageRank {
	
	public static int RESIDUAL_OFFSET = 1000000000;
	public static double DAMPING_FACTOR = 0.85;
	public static int NUM_NODES = 685229;
	
	public static enum COUNTERS {
		RESIDUAL_SUM,
		NUM_RESIDUALS
	};

    public static class Map extends MapReduceBase implements Mapper<LongWritable, Text, IntWritable, Text> {
    	
    	//mapper gets <a b PR(a)>
    	public void map(LongWritable key, Text value, OutputCollector<IntWritable, Text> output, Reporter reporter) throws IOException {
    		System.out.println("mapper got key " + key + " and value " + value);
    		String line = value.toString();
    		String[] inputs = line.split(" ");
    		try {
    			IntWritable firstKey = new IntWritable(Integer.parseInt(inputs[0]));
    			IntWritable secondKey = new IntWritable(Integer.parseInt(inputs[1]));
    			output.collect(firstKey, value);
    			output.collect(secondKey, value);
    		} catch (Exception e){
    			System.out.println("mapper got invalid format");
    			e.printStackTrace();
    		}
    		//reporter.incrCounter(COUNTERS.NUM_NODES, 1);
    	}
    }
    
    private static class EdgeInfo {
    	public int fromNode;
    	public int toNode;
    	public float fromNodePR;
    	public int fromNodeDegree;
    	
    	public EdgeInfo(int fromNode, int toNode, float fromNodePR, int fromNodeDegree){
    		this.fromNode = fromNode;
    		this.toNode = toNode;
    		this.fromNodePR = fromNodePR;
    		this.fromNodeDegree = fromNodeDegree;
    	}
    }

    public static class Reduce extends MapReduceBase implements Reducer<IntWritable, Text, IntWritable, Text> {
    	
    	//reduce gets <a, a b PR(a) PR(b) deg(a)>
    	public void reduce(IntWritable key, Iterator<Text> values, OutputCollector<IntWritable, Text> output, Reporter reporter) throws IOException {
    		System.out.println("reducer got key " + key);
    		Set<EdgeInfo> inSet = new HashSet<EdgeInfo>();
    		Set<EdgeInfo> outSet = new HashSet<EdgeInfo>();
    		while (values.hasNext()){
    			Text value = values.next();
    			System.out.println("\t and value " + value.toString());
    			String[] split = value.toString().split(" ");
    			EdgeInfo edgeInfo = new EdgeInfo(Integer.parseInt(split[0]), Integer.parseInt(split[1]), 
    					Float.parseFloat(split[2]), Integer.parseInt(split[3]));
    			if (key.get() == Integer.parseInt(split[0])){
    				outSet.add(edgeInfo);
    			} else {
    				inSet.add(edgeInfo);
    			}
    		}
    		float newPR = 0;
    		for (EdgeInfo entry : inSet){
    			newPR = newPR + (entry.fromNodePR / entry.fromNodeDegree);
    			//System.out.println("incremented new pagerank by " + (entry.fromNodePR / entry.fromNodeDegree));
    			//System.out.println("fromNodePR is " + entry.fromNodePR + " and fromNodeDegree is " + entry.fromNodeDegree);
    		}
    		System.out.println("got NUM_NODES counter: " + NUM_NODES);
    		
    		//add damping factor
    		newPR = (float) (((1-DAMPING_FACTOR)/NUM_NODES) + (DAMPING_FACTOR*newPR));
    				
    		float oldPR = -1;
    		for (EdgeInfo entry : outSet){
    			output.collect(null, new Text("" + entry.fromNode + " " + entry.toNode  + " " + newPR + " " + entry.fromNodeDegree));
    			oldPR = entry.fromNodePR;
    		}
    		System.out.println("reducer has computed new pageRank " + newPR + " for node " + key.get());
    		float residual = Math.abs((oldPR - newPR)/newPR);
    		System.out.println("reducer has computed residual " + residual);
    		reporter.incrCounter(COUNTERS.RESIDUAL_SUM, (int)(residual*RESIDUAL_OFFSET));
    		reporter.incrCounter(COUNTERS.NUM_RESIDUALS, 1);
    		//System.out.println("size of inSet is " + inSet.size() + " and size of outSet is " + outSet.size());
    	}
    }

    public static void main(String[] args) throws Exception {
    	
    	for (int i = 0; i < 5; i++){
    		System.out.println("-----------------------RUNNING PASS " + i + "-------------------------");
    		JobConf conf = new JobConf(SimplePageRank.class);
	    	conf.setJobName("simplepagerank");
	
	    	conf.setOutputKeyClass(IntWritable.class);
	    	conf.setOutputValueClass(Text.class);
	
	    	conf.setMapperClass(Map.class);
	    	//conf.setCombinerClass(Combine.class);
	    	conf.setReducerClass(Reduce.class);
	
	    	conf.setInputFormat(TextInputFormat.class);
	    	conf.setOutputFormat(TextOutputFormat.class);
	      
	      
    	  
	    	//FileInputFormat.setInputPaths(conf, new Path("/home/ben/Documents/5300/hadoop_io_3/temp/file" + i));
	    	//FileOutputFormat.setOutputPath(conf, new Path("/home/ben/Documents/5300/hadoop_io_3/temp/file" + (i+1)));
    	  
	    	FileInputFormat.setInputPaths(conf, new Path(args[0] + i));
	    	FileOutputFormat.setOutputPath(conf, new Path(args[0] + (i+1)));
	    	System.out.println("input path is " + args[0] + i);
	    	System.out.println("output path is " + args[0] + (i+1));
    	  
	    	RunningJob job = JobClient.runJob(conf);
	    	Counters counters = job.getCounters();
	    	float residualSum = ((float)counters.getCounter(COUNTERS.RESIDUAL_SUM))/RESIDUAL_OFFSET;
	    	float numResiduals = counters.getCounter(COUNTERS.NUM_RESIDUALS);
	    	float residualAverage = residualSum / numResiduals;
	    	System.out.println("Residual sum for this pass: " + residualSum);
	    	System.out.println("Average residual for this pass: " + residualAverage);
      }
    }
}


