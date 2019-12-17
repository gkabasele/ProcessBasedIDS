package algorithms;

import java.io.*;
import java.net.URL;
import java.util.List;
import java.util.Map;

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
		//String transactions = fileToPath("/textfiles/test_close_transactions.txt");
		String transactions = fileToPath("/textfiles/database.txt");
		//String frequentItemsets = ".//test_frequent.txt";
		String frequentItemsets = ".//output.txt";
		//String filtered_output = ".//test_close_frequent_output.txt";
		String filtered_output = ".//output_filtered";
		//String MISfile = fileToPath("/textfiles/test_mis_close.txt");
		String MISfile = fileToPath("/textfiles/mis_v2.txt");
		String closeOutput = ".//close_output_map.txt";
		String invariants = ".//invariants.txt";

		// Check how to input is organised to determine if it need some prepocessing
		/*
		InputVerifier verifier = new InputVerifier();
		verifier.verifyInput(frequentItemsets);
		System.out.println(verifier);
		*/

		// STEP 1: Applying the CFP-GROWTH algorithm to find frequent itemsets
		//AlgoCFPGrowth cfpgrowth = new AlgoCFPGrowth();
		//cfpgrowth.runAlgorithm(transactions, frequentItemsets, MISfile);

		//double minsup =  0.1 * cfpgrowth.getDatabaseSize();
		//cfpgrowth.printStats();

		AssociationRuleMining miner = new AssociationRuleMining();
		//System.out.println("Filtering frequent request with minsup: " + minsup);
		//miner.filterItemSets(frequentItemsets, filtered_output, minsup);

		System.out.println("Mining the invariants");

		// Approach 1 : Keep a list of the close itemset so far and look with new itemset if list must be updated
		//miner.fillCloseItemSet(frequentItemsets);
		//miner.exportCloseItemsets(closeOutput);


		// Approach 2 : Store every itemset then look which one are closed
		Map<Byte, List<ItemSet>> map = miner.miningRules(frequentItemsets);
		miner.exportCloseItemsets(closeOutput, map);

		System.out.println("Exporting invariants");
		miner.exportRule(invariants);
		System.out.println("Done");
	}
	
	public static String fileToPath(String filename) throws UnsupportedEncodingException{
		URL url = MainTestAllAssociationRules_CFPGrowth_saveToFile.class.getResource(filename);
		 return java.net.URLDecoder.decode(url.getPath(),"UTF-8");
	}
}
