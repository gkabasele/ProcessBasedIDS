package  algorithms;
import com.sun.corba.se.spi.ior.IORTemplate;

import java.awt.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.List;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.Map.Entry;
import java.lang.Integer;

import java.io.*;

import static java.nio.file.StandardOpenOption.APPEND;
import static java.nio.file.StandardOpenOption.CREATE;

class ItemSet implements Serializable {
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
	// Items ID -> Items
	private Map<Short, Short> itemsMap;

	public static final short NOTFOUND = -1;
	public static final int CHACHESIZE = 1000000;
	// Constant the differentiate case where a the support of the cause
	// is the same as the support of the effect
	public static final byte SAMESUP = 1;
	public static final byte GREATERSUP = 2;
	public static final byte LESSERSUP = 3;
	public static final byte CONTINUE = 4;

	public AssociationRuleMining(){
		rules = new ArrayList<>();
		closeItemsets = new ArrayList<>();
		itemsMap = new HashMap<>();
	}

	public List<ItemSet> getCloseItemsets() {
		return closeItemsets;
	}

	public void fillCloseItemSetFromFreq(String input) throws IOException{
		// Read File line by line

		String line = null;
		ItemSet itemset = null;
		List<ItemSet> candidates = new ArrayList<>();
		int i = 0;
		long startTime = System.nanoTime();
		try (FileInputStream inputStream = new FileInputStream(input); Scanner sc = new Scanner(inputStream, "UTF-8")) {
			while (sc.hasNextLine()) {
				// Itemsets creation from line
				line = sc.nextLine();
				itemset = stringToItemSet(line);
				candidates = updateCandidates(itemset, candidates);

				i += 1;
				if (i % 10000 == 0) {
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
		candidates.sort(Comparator.comparingInt(ItemSet::getSupport));
		closeItemsets =  candidates;
	}

	public void importItemsSupport(String support) throws IOException {
		String line;
		String[] values;
		short itemsID;
		short itemsSupport;
		try(FileInputStream inputStream = new FileInputStream(support); Scanner sc = new Scanner(inputStream, "UTF-8")){
			while(sc.hasNextLine()){
				line = sc.nextLine();
				values = line.split(" ");
				itemsID = Short.parseShort(values[0]);
				itemsSupport = Short.parseShort(values[1]);
				itemsMap.put(itemsID, itemsSupport);
			}
		}
	}

	public void importCloseItemSets (String closeInput) throws IOException {
		String line;
		ItemSet itemSet;
		try(FileInputStream inputStream = new FileInputStream(closeInput); Scanner sc = new Scanner(inputStream, "UTF-8")){
			while (sc.hasNextLine()){
				line = sc.nextLine();
				itemSet  = stringToItemSet(line);
				closeItemsets.add(itemSet);
			}
		}
		closeItemsets.sort(Comparator.comparingInt(ItemSet::getSupport));
	}

	public void writeItemsetsToFileBin(String input, String output){
		String line;
		ItemSet itemSet;
		try{
			FileInputStream inputStream = new FileInputStream(input);
			Scanner sc = new Scanner(inputStream, "UTF-8");

			ObjectOutputStream out = new ObjectOutputStream(Files.newOutputStream(Paths.get(output), CREATE, APPEND));
			while(sc.hasNextLine()){
				line = sc.nextLine();
				itemSet = stringToItemSet(line);
				out.writeObject(itemSet);
			}
			out.close();

		} catch (IOException ie){
			System.err.println("Could not read file: " + output);
		}

	}

	public void readItemsetsFromBin(String input){
		try{
			FileInputStream inputStream = new FileInputStream(input);
			ObjectInputStream in = new ObjectInputStream(inputStream);

			while(true){
				ItemSet itemSet = (ItemSet) in.readObject();
				System.out.println(itemSet);
			}
		} catch (EOFException eof){
			System.out.println("Reach end of file");
		} catch (ClassNotFoundException cle){
			System.err.println("Could not serialize process");
		} catch (IOException e){
			System.err.println("Find could not be found");
		}
	}

	public void miningRulefromClose(String freqInput, boolean binaryFile){
		createRulesFromCloseItemset(freqInput, binaryFile);
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

	private List<ItemSet> updateCandidates(ItemSet itemSet, List<ItemSet> candidates){
		List<ItemSet> list = new ArrayList<>();
		boolean hasSupersetCandidate = false;
		for (ItemSet cand : candidates){
			if (itemSet.isSupersetOf(cand) && itemSet.getSupport() == cand.getSupport()){
				continue;
			} else if (itemSet.isSubsetOf(cand) && itemSet.getSupport() == cand.getSupport()){
				hasSupersetCandidate = true;
				list.add(cand);
			} else {
				list.add(cand);
			}
		}

		if (!hasSupersetCandidate){
			list.add(itemSet);
		}
		return list;
	}

	private ItemSet  stringToItemSet(String line){
		String[] values = line.split(":");
		short support = Short.parseShort(values[1].replaceAll("\\s", ""));
		String[] it = values[0].split("#");
		String[] itemsID = it[0].split(" ");
		Set<Short> items = new LinkedHashSet<>(itemsID.length);
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
		byte currentSize;
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

	private void createRulesFromCloseItemset(String filename, boolean binaryFile){
		long startTime = System.nanoTime();
		LinkedHashMap<Set<Short>, Short> cache = new LinkedHashMap<Set<Short>, Short>(){
			@Override
			protected boolean removeEldestEntry(Entry entry) {
				return size() > CHACHESIZE;
			}
		};
		int index = 1;
		for (ItemSet itemSet : closeItemsets){
			System.out.println("Starting close itemset: (" + index + "/" + closeItemsets.size() + ")");
			if (isRelevantCloseItemSet(itemSet)){
				rulesFromSet(itemSet, filename, cache, binaryFile);
			} else{
				System.out.println("Irrelevant close itemset");
			}
			index += 1;
		}
		double duration = (System.nanoTime() - startTime)/100000000;
		System.out.println("Duration (s): " + duration);
	}

	// Assume that the itemset is sorted by the support of its items
	private boolean isRelevantCloseItemSet(ItemSet itemSet) {
		List<Short> itemsIDList = new ArrayList<>(itemSet.getItems());
		if (itemSet.length() < 2)
		    return false;
		return (itemsMap.get(itemsIDList.get(0)) == itemSet.getSupport() && itemsMap.get(itemsIDList.get(1)) == itemSet.getSupport());
	}


	private boolean detectCloseItemset(byte size, byte maxKey, ItemSet itemset, Map<Byte, List<ItemSet>> map){
		boolean hasSuperset;
		boolean isClose;
		for (byte supersetSize = (byte)(size + 1); supersetSize <= maxKey; supersetSize++){
			List<ItemSet> list = map.get(supersetSize);
			if (list != null){
				hasSuperset = false;
				for (ItemSet candidateSuperset : list){
					if(candidateSuperset.isSupersetOf(itemset)){
						hasSuperset = true;
						isClose = candidateSuperset.getSupport() != itemset.getSupport();
						if (! isClose){
							return false;
						}
					}
				}
				if (!hasSuperset){
					return true;
				}
			}
		}
		return true;
	}

	private byte checkSupport(short causeSupport, short effectSupport){
		if (causeSupport == effectSupport){
			return SAMESUP;
		} else if (causeSupport > effectSupport){
			return GREATERSUP;
		} else {
			return LESSERSUP;
		}
	}

	private byte sameSupport(Set<Short> cause, Set<Short> effect, String filename,
								LinkedHashMap<Set<Short>, Short> cache, boolean binaryFile) throws IOException{
		short causeSupport = -1;
		short effectSupport = -1;

		if (cache.containsKey(cause))
			causeSupport = cache.get(cause);

		if (cache.containsKey(effect))
			effectSupport = cache.get(effect);

		if (causeSupport > -1 && effectSupport > -1)
			return checkSupport(causeSupport, effectSupport);

		if(binaryFile)
			return findCauseEffectBin(cause, effect, filename, cache);
		else
			return findCauseEffectText(cause, effect, filename, cache);
	}

	private byte findCauseEffectBin(Set<Short> cause, Set<Short>effect, String filename,
									LinkedHashMap<Set<Short>, Short> cache){
		short causeSupport = -1;
		short effectSupport = -1;
		try{
			FileInputStream inputStream = new FileInputStream(filename);
			ObjectInputStream in = new ObjectInputStream(inputStream);

			while(true){
				ItemSet itemSet = (ItemSet) in.readObject();
				cache.putIfAbsent(itemSet.getItems(), itemSet.getSupport());
				if(itemSet.getItems().equals(cause)){
					causeSupport = itemSet.getSupport();
				} else if (itemSet.getItems().equals(effect)){
					effectSupport = itemSet.getSupport();
				}

				if (causeSupport > -1 && effectSupport > -1){
					inputStream.close();
					return checkSupport(causeSupport, effectSupport);
				}
			}
		} catch (EOFException eof){
		    return CONTINUE;
		} catch (ClassNotFoundException cle){

		} catch (IOException e){

		}
		return CONTINUE;
	}

	private byte findCauseEffectText(Set<Short> cause, Set<Short>effect, String filename,
								 LinkedHashMap<Set<Short>, Short> cache) throws IOException{
		String line;
		ItemSet itemset;
		short causeSupport = -1;
		short effectSupport = -1;
		try (FileInputStream inputStream = new FileInputStream(filename); Scanner sc = new Scanner(inputStream, "UTF-8")){
			while (sc.hasNextLine()){
				line = sc.nextLine();
				itemset = stringToItemSet(line);
				cache.putIfAbsent(itemset.getItems(), itemset.getSupport());
				if (itemset.getItems().equals(cause)){
					causeSupport = itemset.getSupport();
				} else if (itemset.getItems().equals(effect)){
					effectSupport = itemset.getSupport();
				}

				if (causeSupport > -1 && effectSupport > -1){
					inputStream.close();
					return checkSupport(causeSupport, effectSupport);
				}
			}
			inputStream.close();
		}

		return CONTINUE;
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

	private byte divideFromLength(Short[] itemSetList, int length, String filename,
									 LinkedHashMap<Set<Short>, Short> cache, boolean binaryFile){
		Short[] cause = new Short[length];
		short root = itemSetList[0];
		return itemSetDivision(itemSetList, root, length, 0, cause, 0, filename, cache, binaryFile);
	}

	private boolean divideFromLength(Short[] itemSetList, int length, Map<Byte, List<ItemSet>> map){
		Short[] cause = new Short[length];
		short root = itemSetList[0];
		return itemSetDivision(itemSetList, root, length, 0, cause, 0, map);
	}

	private List<Short> getEffectFromCause(List<Short> itemSetList, List<Short> cause){
		List<Short> effect = new ArrayList<>();
		for(short items: itemSetList){
			if (! cause.contains(items)){
				effect.add(items);
			}
		}
		return effect;
	}

	private byte  itemSetDivision(Short[] itemSetList, short root, int length, int index, Short[] cause, int i,
									 String filename, LinkedHashMap<Set<Short>, Short> cache, boolean binaryFile){
		byte res;
		if (index == length){
			List<Short> causelist = Arrays.asList(cause);
			List<Short> items = Arrays.asList(itemSetList);
			List<Short> effect = getEffectFromCause(items, causelist);
			try {
				res = createRule(new HashSet<>(causelist), new HashSet<>(effect), filename, cache, binaryFile);
				return res;
			} catch (IOException e) {
				System.out.println("File error for " + causelist + " -> " + effect);
			}
		}

		if (i >= itemSetList.length){
			return CONTINUE;
		}

		cause[index] = itemSetList[i];
		if (cause[0] != root){
			return CONTINUE;
		}
		res = itemSetDivision(itemSetList, root, length, index + 1, cause, i + 1, filename, cache, binaryFile);
		if (res != SAMESUP){
			byte tmp = itemSetDivision(itemSetList, root, length, index, cause, i + 1, filename, cache, binaryFile);
			if (tmp != res && (tmp == SAMESUP || tmp == LESSERSUP)){
				res = tmp;
			}
		}
		return res;
	}

	private boolean itemSetDivision(Short[] itemSetList, short root, int length, int index, Short[] cause, int i,
									Map<Byte, List<ItemSet>> map){
	    boolean res;
		if (index == length){
			List<Short> causelist = Arrays.asList(cause);
			List<Short> items = Arrays.asList(itemSetList);
			List<Short> effect = getEffectFromCause(items, causelist);
			res = createRule(new HashSet<>(causelist), new HashSet<>(effect), map);
			return res;
		}

		if (i >= itemSetList.length){
			return false;
		}

		cause[index] = itemSetList[i];
		if (cause[0] != root){
			return false;
		}
		res = itemSetDivision(itemSetList, root, length, index + 1, cause, i + 1, map);
		if (! res){
		    res = itemSetDivision(itemSetList, root, length, index, cause, i + 1, map);
		}
		return res;
	}

	private byte createRule(Set<Short> cause, Set<Short> effect, String filename,
							   LinkedHashMap<Set<Short>, Short> cache, boolean binaryFile) throws IOException {
		byte same = sameSupport(cause, effect, filename, cache, binaryFile);
		if (same == SAMESUP){
			AssociationRule rule = new AssociationRule(new ArrayList<>(cause), new ArrayList<>(effect));
			System.out.println("Add rules: " + rule);
			rules.add(rule);
		}
		return same;
	}

	private boolean createRule(Set<Short> cause, Set<Short> effect, Map<Byte, List<ItemSet>> map){
		short cisSupport = findSupport(cause, map);
		short eisSupport = findSupport(effect, map);
		if (cisSupport == eisSupport){
			AssociationRule rule = new AssociationRule(new ArrayList<>(cause), new ArrayList<>(effect));
			System.out.println("Add rules: " + rule);
			rules.add(rule);
			return true;
		}
		return false;
	}

	private void rulesFromSet(ItemSet itemset, String filename,
							  LinkedHashMap<Set<Short>, Short> cache, boolean binaryFile){
		byte res;
		Short[] array = new Short[itemset.length()];
		array = (new ArrayList<>(itemset.getItems())).toArray(array);


		for(int i=1; i < itemset.length(); i++){
		    System.out.println("Cause of length  (" + i + "/" + itemset.length()+")");
			res = divideFromLength(array, i, filename, cache, binaryFile);
			if (res == SAMESUP || res == LESSERSUP  ){
				return;
			}
		}
	}

	private void rulesFromSet(ItemSet itemset, Map<Byte, List<ItemSet>> map){
		boolean res;
		Short[] array = new Short[itemset.length()];
		array = (new ArrayList<>(itemset.getItems())).toArray(array);

		for(int i=1; i < itemset.length(); i++){
		   	res = divideFromLength(array, i, map);
			if (res){
				return;
			}
		}
	}
}
