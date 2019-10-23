package algorithms;

import java.io.*;
import java.net.URL;

/**
 * Example of how to mine all association rules with CFPGROWTH and save
 * the result to a file, from the source code.
 * 
 * @author Philippe Fournier-Viger (Copyright 2014)
 */
public class MainTestAllAssociationRules_CFPGrowth_saveToFile {



	public static void main(String [] arg) throws IOException{
		File directory = new File("./");
		System.out.println(directory.getAbsolutePath());
		String input = fileToPath("/textfiles/input.txt");
		String output = ".//output.txt";
		String filtered_output = ".//output_filtered.txt";
		String MISfile = fileToPath("/textfiles/mis.txt");
		String invariants = ".//invariants.txt";
		
		// STEP 1: Applying the CFP-GROWTH algorithm to find frequent itemsets
		AlgoCFPGrowth cfpgrowth = new AlgoCFPGrowth();
		cfpgrowth.runAlgorithm(input, output, MISfile);

		double minsup =  0.1 * cfpgrowth.getDatabaseSize();
		cfpgrowth.printStats();

		AssociationRuleMining miner = new AssociationRuleMining();
		System.out.println("Filtering frequent request with minsup: " + minsup);
		miner.filterItemSets(output, filtered_output, minsup);

		System.out.println("Mining the invariants");
		miner.fillItemSet(filtered_output);
		miner.miningRules();
		miner.exportRule(invariants);
		System.out.println("Done");
	}
	
	public static String fileToPath(String filename) throws UnsupportedEncodingException{
		URL url = MainTestAllAssociationRules_CFPGrowth_saveToFile.class.getResource(filename);
		 return java.net.URLDecoder.decode(url.getPath(),"UTF-8");
	}
}
