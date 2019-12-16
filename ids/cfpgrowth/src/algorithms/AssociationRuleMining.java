package  algorithms;
import com.sun.corba.se.spi.ior.IORTemplate;

import java.util.*;
import java.util.function.Predicate;
import java.util.stream.Collectors;
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

	public boolean containsItem(short item){
		return items.contains(item);
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

	public boolean isSubsetOf(ItemSet itemSet) {return itemSet.isSupersetOf(this);}

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

	private List<ItemSet> closeItemsets;
	private List<AssociationRule> rules;

	public static short NOTFOUND = -1;

	public AssociationRuleMining(){
		rules = new ArrayList<>();
		closeItemsets = new ArrayList<>();
	}

	public List<ItemSet> getCloseItemsets() {
		return closeItemsets;
	}

	public void fillCloseItemSet(String input) throws IOException{
		// Read File line by line

		String line = null;
		ItemSet itemset = null;
		List<ItemSet> candidates = new ArrayList<>();
		Predicate<ItemSet> predicate = null ;
		int i = 0;
		long startTime = System.nanoTime();
		try (FileInputStream inputStream = new FileInputStream(input); Scanner sc = new Scanner(inputStream, "UTF-8")) {
			while (sc.hasNextLine()) {
				// Itemsets creation from line
				line = sc.nextLine();
				itemset = stringToItemSet(line);
				ItemSet temp = itemset;
				predicate = other -> (!(temp.isSupersetOf(other) || temp.isSubsetOf(other)) || (other.getSupport() != temp.getSupport()));
				candidates = candidates.stream().filter(predicate).collect(Collectors.toList());
				candidates.add(itemset);

				i += 1;
				if (i % 5000 == 0) {
					System.out.println("Up to line: " + i);
				}
			}
			if (sc.ioException() != null) {
				throw sc.ioException();
			}
		}
		long endTime = System.nanoTime();
		double duration = (endTime - startTime)/1000000000;
		System.out.println("Duration (s): " + duration);
		closeItemsets = candidates;
	}

	public void exportCloseItemsets(String filename) throws IOException{
		File wfile = new File(filename);

		if (!wfile.exists()){
			wfile.createNewFile();
		}
		BufferedWriter bw = new BufferedWriter((new FileWriter(filename)));
		for(ItemSet itemSet : closeItemsets){
			bw.write(itemSet.toString());
			bw.write("\n");
		}
		bw.close();
	}

	public void exportCloseItemsets(String filename, Map<Byte, List<ItemSet>> map) throws IOException {
		File wfile = new File(filename);
		if (!wfile.exists()){
			wfile.createNewFile();
		}
		BufferedWriter bw = new BufferedWriter((new FileWriter(filename)));
		for (Entry<Byte, List<ItemSet>> entry: map.entrySet()){
			for (ItemSet itemSet : entry.getValue()){
			    if (itemSet.isClose()){
					bw.write(itemSet.toString());
					bw.write("\n");
				}
			}
		}
		bw.close();
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

	private ItemSet  stringToItemSet(String line){
		String[] values = line.split(":");
		short support = Short.parseShort(values[1].replaceAll("\\s", ""));
		String[] it = values[0].split("#");
		String[] itemsID = it[0].split(" ");
		Set<Short> items = new HashSet<>(itemsID.length);
		for (String s: itemsID){
			items.add(Short.parseShort(s));
		}
		return new ItemSet(items, support);
	}

	public Map<Byte, List<ItemSet>> miningRules(String filename) throws IOException {

		File file = new File(filename);
		BufferedReader br = new BufferedReader(new FileReader(file));
		String st;
		ItemSet itemset;
		List<ItemSet> list;
		Map<Byte,List<ItemSet>> map = new TreeMap<>();
		byte maxLength = 0;
		int i = 0;
		long startTime = System.nanoTime();
		while ((st = br.readLine()) != null){
			itemset = stringToItemSet(st);
			byte key = (byte) itemset.getItems().size();
			if (map.containsKey(key)){
				list = map.get(key);
				list.add(itemset);
			} else {
				list = new ArrayList<>();
				list.add(itemset);
				map.put(key, list);
			}
			maxLength = (byte) Math.max(maxLength, itemset.length());
			i += 1;
			if (i% 10000 == 0){
				System.out.println("Up to line: " + i);
			}
		}
		createRulesFromCloseItemset(maxLength, map);
		double duration = (System.nanoTime() - startTime)/100000000;
		System.out.println("Duration (s): " + duration);
		br.close();
		return map;
	}

	private void createRulesFromCloseItemset(byte lastKey, Map<Byte, List<ItemSet>> map){
		byte currentSize = 0;
		boolean isClose;
		for (Entry<Byte, List<ItemSet>> entry: map.entrySet()){
		    List<ItemSet> list = entry.getValue();
		    currentSize = entry.getKey() ;
			for (ItemSet itemSet : list){
				isClose = detectCloseItemset(currentSize, lastKey, itemSet, map);
				itemSet.setClose(isClose);
				if (isClose){
					rulesFromSet(itemSet, map);
				}
			}
		}
	}

	private boolean detectCloseItemset(byte size, byte maxKey, ItemSet itemset, Map<Byte, List<ItemSet>> map){
		boolean hasSuperset;
		boolean isClose = true;
		for (byte supersetSize = (byte)(size + 1); supersetSize < maxKey; supersetSize++){
			List<ItemSet> list = map.get(supersetSize);
			if (list != null){
				hasSuperset = false;
				for (ItemSet candidateSuperset : list){
					if(candidateSuperset.isSupersetOf(itemset)){
						hasSuperset = true;
						isClose = candidateSuperset.getSupport() != itemset.getSupport();
						if (! isClose){
							return isClose;
						}
					}
				}
				if (!hasSuperset){
					return true;
				}
			}
		}
		return isClose;
	}

	private short findSupport(Set<Short> set, Map<Byte, List<ItemSet>> map){
		for (Entry<Byte, List<ItemSet>> entry: map.entrySet()){
		    if(entry.getKey() == set.size()){
				for (ItemSet itemSet : entry.getValue()){
					if (itemSet.getItems().equals(set)){
						return itemSet.getSupport();
					}
				}
			}
		}
		return NOTFOUND;
	}

	private void rulesFromSet(ItemSet itemset, Map<Byte, List<ItemSet>> map){
		for(int i=1; i < itemset.length(); i++){
			Set<Short> cause = new HashSet<>(new ArrayList<>(itemset.getItems()).subList(0,i));
			Set<Short> effect = new HashSet<>(new ArrayList<>(itemset.getItems()).subList(i, itemset.length()));


			short cisSupport = findSupport(cause, map);
			short eisSupport = findSupport(effect, map);

			if (cisSupport == eisSupport){
				AssociationRule rule = new AssociationRule(new ArrayList<>(cause), new ArrayList<>(effect));
				rules.add(rule);
				break;
			}
		}
	}
}