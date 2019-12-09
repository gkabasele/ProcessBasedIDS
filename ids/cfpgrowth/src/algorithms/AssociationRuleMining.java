package  algorithms;
import java.util.*;
import java.util.Map.Entry;
import java.lang.Integer;

import java.io.*;

class ItemSet {
	private Set<Short> items;
	private short support;
	private boolean isClose;

	public ItemSet(Set<Short> items, short support) {
		this.items = items;
		this.support = support;
		this.isClose = false;
	}

	public Set<Short> getItems(){
		return items;	
	}

	public short getSupport(){
		return support;	
	}

	public boolean isClose() {
		return isClose;
	}

	public void setClose(boolean isClose){
		this.isClose = isClose;
	}

	public int length(){
		return items.size();	
	}

	@Override
	public int hashCode(){
		return items.hashCode();
	}

	public boolean isSupersetOf(ItemSet itemset){
		return items.containsAll(itemset.getItems());
	}

	public String toString(){
		StringBuilder st = new StringBuilder();
		for (Short i : items){
			st.append(i).append(" ");
		}
		st.append("#SUP: ").append(support);
		return st.toString();
	}

	@Override
	public boolean equals(Object o) {
		if (this == o) return true;
		if (o == null || getClass() != o.getClass()) return false;
		ItemSet itemSet = (ItemSet) o;
		return itemSet.support == support && items.equals(itemSet.items);
	}
}

class AssociationRule{
	private List<Short> cause;
	private List<Short> effect;

	public AssociationRule(List<Short> cause, List<Short> effect){
		this.cause = cause;
		this.effect = effect;	
	}

	public String toString(){
		return cause.toString() + "->" + effect.toString();	
	}
}

public class AssociationRuleMining {

	private TreeMap<Byte,List<ItemSet>> itemsets;
	private List<AssociationRule> rules;

	public static short NOTFOUND = -1;

	public AssociationRuleMining(){
		itemsets = new TreeMap<Byte, List<ItemSet>>();
		rules = new ArrayList<AssociationRule>();
	}

	public void fillItemSet(String input) throws IOException{
		// Read File line by line
		FileInputStream inputStream = null;
		Scanner sc = null;

		String line;
		String [] values;
		short support;
		String[] it;
		String[] itemsID;
		Set<Short> items;
		byte key;
		List<ItemSet> temp;
		ItemSet itemset;
		try{
			inputStream = new FileInputStream(input);
			sc = new Scanner(inputStream, "UTF-8");
			while (sc.hasNextLine()){
					// Itemsets creation from line
					line = sc.nextLine();
					values = line.split(":");
					support = Short.parseShort(values[1].replaceAll("\\s", ""));
					// ["Itemset", "SUP"]
					it = values[0].split("#");
					itemsID = it[0].split(" ");
					items = new HashSet<>(itemsID.length);
					for (String s : itemsID) {
						items.add(Short.parseShort(s));
					}
					itemset = new ItemSet(items, support);
					key = (byte) itemset.length();
					if (itemsets.containsKey(key)) {
						temp = itemsets.get(key);
						temp.add(itemset);
					} else {
						itemsets.put(key, new ArrayList<>(Arrays.asList(itemset)));
					}
			}
			if (sc.ioException() != null){
				throw sc.ioException();
			}
		} finally {
			if (inputStream != null) {
				inputStream.close();
			}
			if (sc != null){
				sc.close();
			}
		}

		 setCloseFrequentItemSets();
	}

	public void exportTreeMap(String output) throws IOException {
		File wfile = new File(output);
		BufferedWriter bw = new BufferedWriter(new FileWriter(wfile));
		for(Entry<Byte, List<ItemSet>> entry : itemsets.entrySet()){
			for(ItemSet itemset : entry.getValue()){
				bw.write(itemset.toString());
				bw.write("\n");
			}
		}
		bw.close();
	}

	public void setCloseFrequentItemSets() {
		for(int i = itemsets.firstKey(); i <= itemsets.lastKey(); i++){
			List<ItemSet> currentList = itemsets.get(i);
			for(ItemSet candidateClose : currentList){
				boolean close = isCloseFrequent(candidateClose, i);
				candidateClose.setClose(close);
			}
		}
	}

	public boolean isCloseFrequent(ItemSet candidate, int index){
		boolean isClose = true;
		for(int j=index+1; j <=itemsets.lastKey(); j++){
			boolean hasSuperset = false;
			List<ItemSet> candidateSuperset = itemsets.get(j);
			if(candidateSuperset != null) {
				for (ItemSet itemSet : candidateSuperset) {
					if (itemSet.isSupersetOf(candidate)) {
						hasSuperset = true;
						if (itemSet.getSupport() == candidate.getSupport()) {
							isClose = false;
							break;
						}
					}
				}
			} else{
				continue;
			}
			if (!isClose || !hasSuperset) {
				break;
			}
		}
		return isClose;
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
		for (Entry<Byte, List<ItemSet>> pair : itemsets.entrySet()) {
			List<ItemSet> itemset = pair.getValue();
			rulesFromSet(itemset);
		}
	}

	private short findSupport(Set<Short> itemSet){
		byte key = (byte) itemSet.size();
		List<ItemSet> itemSetList = itemsets.get(key);
		if (itemSetList != null){
			for(ItemSet cand: itemSetList){
				if (cand.getItems().equals(itemSet)){
					return cand.getSupport();
				}
			}
		}
		return NOTFOUND;
	}

	private void rulesFromSet(List<ItemSet> itemSetList){
		for(ItemSet itemset : itemSetList){
			if(itemset.isClose()){
				for(int i=1; i < itemset.length(); i++){
					Set<Short> cause = new HashSet<>(new ArrayList<>(itemset.getItems()).subList(0,i));
					Set<Short> effect = new HashSet<>(new ArrayList<>(itemset.getItems()).subList(i, itemset.length()));

					short cisSupport = findSupport(cause);
					short eisSupport = findSupport(cause);

					if (cisSupport == eisSupport){
						AssociationRule rule = new AssociationRule(new ArrayList<>(cause), new ArrayList<>(effect));
						rules.add(rule);
						break;
					}
				}
			}
		}
	}
	
	public void run(String input, String outputItemSet, String outputRules) throws Exception {

		AssociationRuleMining miner = new AssociationRuleMining();	
		miner.fillItemSet(input);
		miner.miningRules();
		miner.exportRule(outputRules);
	}
}
