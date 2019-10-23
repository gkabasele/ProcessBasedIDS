package  algorithms;
import java.util.HashMap;
import java.util.Map.Entry;
import java.util.Iterator;
import java.util.ArrayList;
import java.util.List;
import java.lang.Integer;

import java.io.*;

class ItemSet {
	private List<Integer> items;
	private int support;

	public ItemSet(List<Integer> items, int support) {
		this.items = items;
		this.support = support;	
	} 

	public List<Integer> getItems(){
		return items;	
	}

	public int getSupport(){
		return support;	
	}

	public int length(){
		return items.size();	
	}

	@Override
	public int hashCode(){
		return items.hashCode();	
	}
}

class AssociationRule{
	private List<Integer> cause;
	private List<Integer> effect;

	public AssociationRule(List<Integer> cause, List<Integer> effect){
		this.cause = cause;
		this.effect = effect;	
	}

	public String toString(){
		return cause.toString() + "->" + effect.toString();	
	}
}

public class AssociationRuleMining {

	private HashMap<Integer,ItemSet> itemsets;
	private List<AssociationRule> rules;

	public AssociationRuleMining(){
		itemsets = new HashMap<Integer, ItemSet>();
		rules = new ArrayList<AssociationRule>();
	}

	public void fillItemSet(String input) throws IOException{
		File rfile = new File(input);
		BufferedReader br = new BufferedReader(new FileReader(rfile));
		String st;
		while((st = br.readLine()) != null){
			String[] values = st.split(":");
			int support = Integer.parseInt(values[1].replaceAll("\\s", ""));
			// ["Itemset", "SUP"]
			String[] it = values[0].split("#");
			String[] itemsID = it[0].split(" ");
			List<Integer> items = new ArrayList<Integer>(itemsID.length);
			for(String s: itemsID){
				items.add(Integer.parseInt(s));	
			}
		   		
			ItemSet itemset = new ItemSet(items, support);
			addItemSet(itemset);
		}
		br.close();

	}

	public static void filterItemSets(String input, String output, double minsup) throws IOException{
		File rfile = new File(input);
		BufferedReader br = new BufferedReader(new FileReader(rfile));

		File wfile = new File(output);
		BufferedWriter bw = new BufferedWriter(new FileWriter(wfile));

		String st;
		while((st = br.readLine()) != null){
			String[] values = st.split(":");
			int support = Integer.parseInt(values[1].replaceAll("\\s", ""));
			if(support >= minsup){
				bw.write(st);
				bw.write("\n");
			}
		}
		bw.close();
		br.close();
	}

	public void exportRule(String filename) throws IOException{
		File wfile = new File(filename);

		if(!wfile.exists()){
			wfile.createNewFile();	
		}
		BufferedWriter bw = new BufferedWriter(new FileWriter(filename));

		for(AssociationRule rule: rules){
			bw.write(rule.toString());	
			bw.write("\n");
		}
		bw.close();
	}

	public void miningRules(){
		Iterator<Entry<Integer, ItemSet>> it = itemsets.entrySet().iterator();
		while(it.hasNext()){
			Entry<Integer, ItemSet> pair = it.next();
			ItemSet itemset = pair.getValue();	
			rulesFromSet(itemset);
		}
	}

	private void rulesFromSet(ItemSet itemset){
		for(int i=1; i < itemset.length(); i++){
			List<Integer> cause = new ArrayList<Integer>(itemset.getItems().subList(0,i));
			List<Integer> effect = new ArrayList<Integer>(itemset.getItems().subList(i, itemset.length()));

			ItemSet cis = itemsets.get(cause.hashCode());
			ItemSet eis = itemsets.get(effect.hashCode());

			if( cis != null && eis != null){
				if (cis.getSupport() == eis.getSupport()){
					AssociationRule rule = new AssociationRule(cause, effect);
					rules.add(rule);
					break;			
				}	
			}
		}
	}
	
	
	private void addItemSet(ItemSet itemset){
		itemsets.put(itemset.hashCode(), itemset);	
	}

	public static void print(String msg){
		System.out.println(msg);
	}

	public void run(String input, String output) throws Exception {

		AssociationRuleMining miner = new AssociationRuleMining();	
		miner.fillItemSet(input);
		miner.miningRules();
		miner.exportRule(output);
			
	}
}
