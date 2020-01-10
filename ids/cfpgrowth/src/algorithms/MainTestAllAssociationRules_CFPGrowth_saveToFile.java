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


	public static void getFrequentItemsets(String transactions, String MIS, String output) throws IOException {
		AlgoCFPGrowth cfpGrowth = new AlgoCFPGrowth();
		cfpGrowth.runAlgorithm(transactions, output, MIS);
		cfpGrowth.printStats();
	}

	public static void approachList(String frequentItemsets, String closeOutput) throws IOException {
		AssociationRuleMining miner = new AssociationRuleMining();
		System.out.println("Looking for close itemset");
		miner.fillCloseItemSetFromFreq(frequentItemsets);
		System.out.println("Exporting close itemset to : " + closeOutput);
		miner.exportCloseItemsets(closeOutput);
	}

	public static void mineRuleFromCloseItemSets(String frequentItemsets, String support, String closeOutput,
												 String invariants, boolean binaryFile) throws IOException {
		AssociationRuleMining miner = new AssociationRuleMining();
		miner.importItemsSupport(support);
		System.out.println("Importing close itemsets from: " + closeOutput);
		miner.importCloseItemSets(closeOutput);
		System.out.println("Mining rule from using support from: " + frequentItemsets);
		miner.miningRulefromClose(frequentItemsets, binaryFile);
		System.out.println("Exporting invariants to : " + invariants);
		miner.exportRule(invariants);
	}

	public static void approachMap(String frequentItemsets, String closeOutput, String invariants) throws IOException {
		AssociationRuleMining miner = new AssociationRuleMining();
		Map<Byte, List<ItemSet>> map = miner.miningRules(frequentItemsets);
		miner.exportCloseItemsets(closeOutput, map);
		System.out.println("Exporting invariants");
		miner.exportRule(invariants);
	}

	public static void main(String [] arg) throws IOException{
		File directory = new File("./");
		System.out.println(directory.getAbsolutePath());
		String transactions = fileToPath("/textfiles_swat_process/database.txt");
		String frequentItemsetsComplete = ".//output_swat.txt";
		String MISfileComplete = fileToPath("/textfiles_swat_process/mis.txt");
		//String closeOutputComplete = ".//close_output_swat_complete_sorted.txt";
		//String invariants = ".//invariants.txt";

		// Check how to input is organised to determine if it need some prepocessing
		/*
		InputVerifier verifier = new InputVerifier();
		verifier.verifyInput(frequentItemsets);
		System.out.println(verifier);
		*/

		// STEP 1: Applying the CFP-GROWTH algorithm to find frequent itemsets
		getFrequentItemsets(transactions, MISfileComplete, frequentItemsetsComplete);

		//approachList(frequentItemsetsComplete, closeOutputComplete);

		/*
		AssociationRuleMining miner = new AssociationRuleMining();
		String binFile = ".//output_complete.bin";
		File f = new File(binFile);
		if(f.delete()){
			System.out.println("deleting file " + binFile);
		}
		*/

		// Approach 1 : Keep a list of the close itemset so far and look with new itemset if list must be updated
		//miner.writeItemsetsToFileBin(frequentItemsetsComplete, binFile);
		//mineRuleFromCloseItemSets(frequentItemsetsComplete, MISfileComplete, closeOutputComplete, invariantsComplete, false);

		// Approach 2 : Store every itemset then look which one are closed
		//approachMap(frequentItemsets, closeOutput, invariants);

		System.out.println("Done");
	}
	
	public static String fileToPath(String filename) throws UnsupportedEncodingException{
		URL url = MainTestAllAssociationRules_CFPGrowth_saveToFile.class.getResource(filename);
		 return java.net.URLDecoder.decode(url.getPath(),"UTF-8");
	}
}
