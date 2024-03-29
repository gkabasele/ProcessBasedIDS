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
	private int support;
	private boolean isClose;

	public ItemSet(Set<Short> items, int support) {
		this.items = items;
		this.support = support;
		this.isClose = false;
	}

	public Set<Short> getItems(){
		return items;	
	}

	public int getSupport(){
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

class SortBySupport implements Comparator<Short>
{

	private Map<Short, Integer> itemap;

	public SortBySupport(Map<Short, Integer> map){
		this.itemap = map;
	}

	@Override
	public int compare(Short itemID, Short t1) {
		int supportA = itemap.get(itemID);
		int supportB = itemap.get(t1);
		return  supportA - supportB;
	}
}

class AssociationRule{
	private List<Short> cause;
	private List<Short> effect;

	public AssociationRule(List<Short> cause, List<Short> effect){
		this.cause = cause;
		this.effect = effect;	
	}

	public AssociationRule(String line){
		String[] parts = line.split("->");
		String s = parts[0].replace("[", "");
		String first = s.replace("]", "");
		s = parts[1].replace("[", "");
		String second = s.replace("]", "");
		String[] sCause = first.split(",");
		String[] sConseq = second.split(",");
		cause = new ArrayList<>();
		effect = new ArrayList<>();
		int biggest = Math.max(sCause.length, sConseq.length);
		for (int i = 0; i < biggest; i++){
			if (i < sCause.length){
				cause.add(Short.parseShort(sCause[i].replace(" ", "")));
			}
			if (i < sConseq.length){
				effect.add(Short.parseShort(sConseq[i].replace(" ", "")));
			}
		}
	}

	public List<Short> getCause() {
		return cause;
	}

	public List<Short> getEffect() {
		return effect;
	}

	public boolean causeSuperserOf(AssociationRule other){
		List<Short> tmpCause = getCause();
		List<Short> tmpOtherCause = other.getCause();

		tmpCause.sort(Comparator.comparingInt(aShort -> aShort));
		tmpOtherCause.sort(Comparator.comparingInt(aShort -> aShort));
		Short[] causeArray = new Short[tmpCause.size()];
		Short[] otherCauseArray = new Short[tmpOtherCause.size()];
		return ArraysAlgos.includedIn(tmpOtherCause.toArray(otherCauseArray), tmpCause.toArray(causeArray));
	}

	public String toString(){
		return cause.toString() + "->" + effect.toString();	
	}

}

public class AssociationRuleMining {

	private List<ItemSet> closeItemsets;
	private List<AssociationRule> rules;
	// Items ID -> Items
	private Map<Short, Integer> itemsMap;

	//Database of transactions
	private List<Short[]> database;

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
		database = new ArrayList<>();
	}

	public List<ItemSet> getCloseItemsets() {
		return closeItemsets;
	}

	/*
	* Fill database with all the transaction
	*/
	public void fillDatabase(String input) throws IOException{
	    System.out.println("Reading the transaction to fill the database");
	    String line;
	    int i = 0;
		try (FileInputStream inputStream = new FileInputStream(input); Scanner sc = new Scanner(inputStream, "UTF-8")) {
			while (sc.hasNextLine()) {
				line = sc.nextLine();
				String itemsID[] = line.split(" ");
				Short items[] = new Short[itemsID.length];
				for (int j = 0; j < itemsID.length ; j++){
				    items[j] = Short.parseShort(itemsID[j]);
				}
				database.add(items);
				i += 1;
				if (i % 10000 == 0) {
					System.out.println("Up to line: " + i);
				}
			}
		}
	}

	/*
	 *	Compute closeItemsets list from a text file containing the frequent itemset
	 */
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

	/*
	*	Fill the support for each items from a text a textfile
	**/
	public void importItemsSupport(String support) throws IOException {
		String line;
		String[] values;
		short itemsID;
		int itemsSupport;
		try(FileInputStream inputStream = new FileInputStream(support); Scanner sc = new Scanner(inputStream, "UTF-8")){
			while(sc.hasNextLine()){
				line = sc.nextLine();
				values = line.split(" ");
				itemsID = Short.parseShort(values[0]);
				itemsSupport = Integer.parseInt(values[1]);
				itemsMap.put(itemsID, itemsSupport);
			}
		}
	}

	/*
		Fill the closeItemsets list from a text file containing the close itemsets
	 */
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

	/*
		Convert a text file of itemset to a binary file
	 */
	public void writeItemsetsToFileBin(String input, String output){
		String line;
		ItemSet itemSet;
		int i = 1;
		try{
			FileInputStream inputStream = new FileInputStream(input);
			Scanner sc = new Scanner(inputStream, "UTF-8");

			ObjectOutputStream out = new ObjectOutputStream(Files.newOutputStream(Paths.get(output), CREATE, APPEND));
			while(sc.hasNextLine()){
				line = sc.nextLine();
				itemSet = stringToItemSet(line);
				out.writeObject(itemSet);
				i++;
				if (i % 100000 == 0){
					System.out.println("Up to line: " + i + " flushing");
					out.flush();
				}
			}
			out.close();

		} catch (IOException ie){
			System.err.println("Could not read file: " + output);
		}

	}

	/*
		Read a binary file and print the itemset contained in that file
	 */
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


	/*
		Write close itemsets to a file from the list closeItemsets
	 */
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

	/*
	* Write close itemsets to a file from a dictonnary containing itemset with  a flag indicating whether there are
	* close or not
	* */
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

	/*
	* Write Invariant rule to a file
	*/
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


	/*
		Remove non closeItemset from the list of candidate based on current itemsets
	 */
	private List<ItemSet> updateCandidates(ItemSet itemSet, List<ItemSet> candidates){
		List<ItemSet> list = new ArrayList<>();
		boolean hasSupersetCandidate = false;
		for (ItemSet cand : candidates) {
			if (itemSet.isSupersetOf(cand) && itemSet.getSupport() == cand.getSupport()) {
				continue;
			} else if (itemSet.isSubsetOf(cand) && itemSet.getSupport() == cand.getSupport()) {
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

	/*
		Create a new itemSet from a string
	 */
	private ItemSet  stringToItemSet(String line){
		String[] values = line.split(":");
		int support = Integer.parseInt(values[1].replaceAll("\\s", ""));
		String[] it = values[0].split("#");
		String[] itemsID = it[0].split(" ");
		Set<Short> items = new LinkedHashSet<>(itemsID.length);
		for (String s: itemsID){
			items.add(Short.parseShort(s));
		}
		return new ItemSet(items, support);
	}

	public void miningRulefromClose(String freqInput, boolean binaryFile){
		createRulesFromCloseItemset(freqInput, binaryFile);
	}

	public void miningRuleFromClose(){
		createRulesFromCloseItemset();
	}

	/*
	* Create a map where the key is the size of an itemSet and the value is a a list the itemsets with size key
	* */
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

	/*
		Create rule when a close itemset has been detected. A map is used for that
		containing all the itemset is used to speed the process. It requires that there is
		enough memory to store all the itemsets
	 */
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

	/*
		Create rule when a close itemset has been detected. A file is need to store all
		the itemsets and there support. A cache is used to speed up the process but since
		the file must be read a lot of time, it is Extremely slow
		filename is the name of the file containing the frequent itemset
	 */
	private void createRulesFromCloseItemset(String filename, boolean binaryFile){
		long startTime = System.nanoTime();
		LinkedHashMap<Set<Short>, Integer> cache = new LinkedHashMap<Set<Short>, Integer>(){
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

	private void createRulesFromCloseItemset(){
		long startTime = System.nanoTime();
		LinkedHashMap<Set<Short>, Integer> cache = new LinkedHashMap<Set<Short>, Integer>(){
			@Override
			protected boolean removeEldestEntry(Entry entry) {
				return size() > CHACHESIZE;
			}
		};
		int index = 1;
		for (ItemSet itemSet : closeItemsets){
			System.out.println("Starting close itemset: (" + index + "/" + closeItemsets.size() + ")");
			rulesFromSet(itemSet, cache);
			index += 1;
		}
		double duration = (System.nanoTime() - startTime)/100000000;
		System.out.println("Duration (s): " + duration);
	}

	private int binomialCoef(int n, int k){
		int C[][] = new int[n+1][k+1];
		int i, j;

		// Calculate  value of Binomial Coefficient in bottom up manner
		for (i = 0; i <= n; i++)
		{
			for (j = 0; j <= Integer.min(i, k); j++)
			{
				// Base Cases
				if (j == 0 || j == i)
					C[i][j] = 1;

					// Calculate value using previously stored values
				else
					C[i][j] = C[i-1][j-1] + C[i-1][j];
			}
		}
		return C[n][k];
	}

	// Returns count of different partitions of n elements in K subsets
	private int countPartitionsWays(int n, int k){
		// Table to store results of subproblems
		int[][] dp = new int[n+1][k+1];

		// Base cases
		for (int i = 0; i <= n; i++)
			dp[i][0] = 0;
		for (int i = 0; i <= k; i++)
			dp[0][k] = 0;

		// Fill rest of the entries in dp[][]
		// in bottom up manner
		for (int i = 1; i <= n; i++)
			for (int j = 1; j <= k; j++)
				if (j == 1 || i == j)
					dp[i][j] = 1;
				else
					dp[i][j] = j * dp[i - 1][j] + dp[i - 1][j - 1];

		return dp[n][k];
	}

	/*
		Look if it is possible to extract an invariant rule from a close itemset
		To generate association rule, there must be at least two itemset with the same support
		such that one of them is in the cause and the other one in the consequence
	 */
	// Assume that the itemset is sorted by the support of its items
	private boolean isRelevantCloseItemSet(ItemSet itemSet) {
		List<Short> itemsIDList = new ArrayList<>(itemSet.getItems());
		if (itemSet.length() < 2)
		    return false;
		itemsIDList.sort(new SortBySupport(itemsMap));
		return (itemsMap.get(itemsIDList.get(0)) == itemsMap.get(itemsIDList.get(1)));
	}

	/*
		Check if an itemset is closed or not, given the list of the other itemsets and their length
	 */
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

	private byte checkSupport(int causeSupport, int effectSupport){
		if (causeSupport == effectSupport){
			return SAMESUP;
		} else if (causeSupport > effectSupport){
			return GREATERSUP;
		} else {
			return LESSERSUP;
		}
	}

	/*
		Look in the binary file to get the support of the two itemsets composing the invariants. The lookup is done
		linearly so a cache is used to avoid lookup as much as possible.
	 */
	private byte sameSupport(Set<Short> cause, Set<Short> effect, String filename,
								LinkedHashMap<Set<Short>, Integer> cache, boolean binaryFile) throws IOException{
		int causeSupport = -1;
		int effectSupport = -1;

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

	private int[] sameSupport(Set<Short> cause, Set<Short> effect,
							 LinkedHashMap<Set<Short>, Integer> cache){
		int causeSupport = -1;
		int effectSupport = -1;

		if (cache.containsKey(cause))
			causeSupport = cache.get(cause);

		if (cache.containsKey(effect))
			effectSupport = cache.get(effect);

		if (causeSupport == -1 && cause.size() == 1 && itemsMap.containsKey((Short) cause.toArray()[0]))
			causeSupport = itemsMap.get((Short) cause.toArray()[0]);

		if (effectSupport == -1 && effect.size() == 1 && itemsMap.containsKey((Short) effect.toArray()[0]))
			effectSupport = itemsMap.get((Short) effect.toArray()[0]);

		if (causeSupport > -1 && effectSupport > -1)
			return new int[]{causeSupport, effectSupport};

		int supports[] = getItemsetSupFromDB(cause, effect);
		cache.put(cause, supports[0]);
		cache.put(effect, supports[1]);
		return supports;

	}
	/*
		Lookup of support of the two itemsets composing invariant on a binary file
	* */
	private byte findCauseEffectBin(Set<Short> cause, Set<Short>effect, String filename,
									LinkedHashMap<Set<Short>, Integer> cache){
		int causeSupport = -1;
		int effectSupport = -1;
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

	/*
		Lookup of support of the two itemsets composing invariant on a text file
	 */
	private byte findCauseEffectText(Set<Short> cause, Set<Short>effect, String filename,
								 LinkedHashMap<Set<Short>, Integer> cache) throws IOException{
		String line;
		ItemSet itemset;
		int causeSupport = -1;
		int effectSupport = -1;
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

	/*
		Lookup of support done via the map containing all the itemsets. To use if memory big enoug to store all itemsets
	 */
	private int findSupport(Set<Short> set, Map<Byte, List<ItemSet>> map){
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

	/*
	*	Generate possible invariant by subdividing itemset in cause and effect. File version
	* */
	private byte divideFromLength(Short[] itemSetList, int length, String filename,
									 LinkedHashMap<Set<Short>, Integer> cache, boolean binaryFile){
		Short[] cause = new Short[length];
		short root = itemSetList[0];
		return itemSetDivision(itemSetList, root, length, 0, cause, 0, filename, cache, binaryFile);
	}

	private byte divideFromLength(ItemSet itemSet,Short[] itemSetList, int length,
								  LinkedHashMap<Set<Short>, Integer> cache){
		Short[] cause = new Short[length];
		return itemSetDivision(itemSet, itemSetList, length, 0, cause, 0, cache);
	}


	/*
	 *	Generate possible invariant by subdividing itemset in cause and effect. Map version
	 * */
	private boolean divideFromLength(Short[] itemSetList, int length, Map<Byte, List<ItemSet>> map){
		Short[] cause = new Short[length];
		short root = itemSetList[0];
		return itemSetDivision(itemSetList, root, length, 0, cause, 0, map);
	}

	/**
	 * Given an itemset and the cause part, return the effect part
	 * */
	private List<Short> getEffectFromCause(List<Short> itemSetList, List<Short> cause){
		List<Short> effect = new ArrayList<>();
		for(short items: itemSetList){
			if (! cause.contains(items)){
				effect.add(items);
			}
		}
		return effect;
	}

	/*
	* Generate of the subdivision with a cause of length @length. Recursive method. File version to get support
	* */
	private byte  itemSetDivision(Short[] itemSetList, short root, int length, int index, Short[] cause, int i,
									 String filename, LinkedHashMap<Set<Short>, Integer> cache, boolean binaryFile){
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

	private byte  itemSetDivision(ItemSet itemSet, Short[] itemSetList, int length, int index, Short[] cause, int i,
								  LinkedHashMap<Set<Short>, Integer> cache){
		byte res;
		if (index == length){
			List<Short> causelist = Arrays.asList(cause);
			List<Short> items = Arrays.asList(itemSetList);
			List<Short> effect = getEffectFromCause(items, causelist);
			res = createRule(itemSet, new HashSet<>(causelist), new HashSet<>(effect), cache);
			return res;
		}

		if (i >= itemSetList.length){
			return CONTINUE;
		}

		cause[index] = itemSetList[i];
		res = itemSetDivision(itemSet, itemSetList, length, index + 1, cause, i + 1, cache);
		if (res != SAMESUP){
			byte tmp = itemSetDivision(itemSet, itemSetList, length, index, cause, i + 1, cache);
			// Should we continue to test next subsets
			if (tmp != res && (tmp == SAMESUP )){
				res = tmp;
			}
		}
		return res;
	}

	/*
	 * Generate of the subdivision with a cause of length @length. Recursive method. Map version to get support
	 * */
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

	/*
	* Create a invariant and add it to the list of invariant if the cause subset and the effect subset have the same
	* support. file version
	* */
	private byte createRule(Set<Short> cause, Set<Short> effect, String filename,
							   LinkedHashMap<Set<Short>, Integer> cache, boolean binaryFile) throws IOException {
		byte same = sameSupport(cause, effect, filename, cache, binaryFile);
		if (same == SAMESUP){
			AssociationRule rule = new AssociationRule(new ArrayList<>(cause), new ArrayList<>(effect));
			System.out.println("Add rules: " + rule);
			rules.add(rule);
		}
		return same;
	}

	private byte createRule(ItemSet itemSet, Set<Short> cause, Set<Short> effect,
							LinkedHashMap<Set<Short>, Integer> cache) {
		int[] supports = sameSupport(cause, effect, cache);
		byte same = checkSupport(supports[0], itemSet.getSupport());
		if (same == SAMESUP){
			AssociationRule rule = new AssociationRule(new ArrayList<>(cause), new ArrayList<>(effect));
			System.out.println("Add rules: " + rule);
			rules.add(rule);
		}

		if (checkSupport(supports[1], itemSet.getSupport()) == SAMESUP){
			AssociationRule rule = new AssociationRule(new ArrayList<>(effect), new ArrayList<>(cause));
			System.out.println("Add rules: " + rule);
			rules.add(rule);
			same = SAMESUP;
		}
		return same;
	}
	/*
	 * Create a invariant and add it to the list of invariant if the cause subset and the effect subset have the same
	 * support. Map version
	 * */
	private boolean createRule(Set<Short> cause, Set<Short> effect, Map<Byte, List<ItemSet>> map){
		int cisSupport = findSupport(cause, map);
		int eisSupport = findSupport(effect, map);
		if (cisSupport == eisSupport){
			AssociationRule rule = new AssociationRule(new ArrayList<>(cause), new ArrayList<>(effect));
			System.out.println("Add rules: " + rule);
			rules.add(rule);
			return true;
		}
		return false;
	}

	/*
	* Generate a rule from the close itemsets if it can be divide in two subsets such both subset have the same support
	* File version.
	* */
	private void rulesFromSet(ItemSet itemset, String filename,
							  LinkedHashMap<Set<Short>, Integer> cache, boolean binaryFile){
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

	private void rulesFromSet(ItemSet itemSet, LinkedHashMap<Set<Short>, Integer> cache){
		byte res;
		Short[] array = new Short[itemSet.length()];
		array = (new ArrayList<>(itemSet.getItems())).toArray(array);


		for(int i=1; i < itemSet.length(); i++){
			System.out.println("Cause of length  (" + i + "/" + itemSet.length()+")");
			res = divideFromLength(itemSet, array, i, cache);
			if (res == SAMESUP){
				return;
			}
		}
	}
	/*
	 * Generate a rule from the close itemsets if it can be divide in two subsets such both subset have the same support
	 * Map version.
	 * */
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

	/*
	*	Return the items with minimum support in an itemset
	* */
	// Assume that the itemset is sorted by the support of its items
	private int getMinSup(ItemSet itemSet){
		List<Short> itemsIDList = new ArrayList<>(itemSet.getItems());
		return (itemsMap.get(itemsIDList.get(0)));
	}

	private int[] getItemsetSupFromDB(Set<Short> cause, Set<Short> effect){
		int supportCause = 0;
		int supportEffect = 0;

		Short causeArray[] = new Short[cause.size()];
		Short effectArray[] = new Short[effect.size()];
		System.arraycopy(cause.toArray(), 0, causeArray, 0, cause.size());
		Arrays.sort(causeArray);
		System.arraycopy(effect.toArray(), 0, effectArray, 0, effect.size());
		Arrays.sort(effectArray);

		int i = 0;
		for (Short transaction[] : database){
			if (itemsetInTransaction(transaction, causeArray))	{
				supportCause += 1;
			}

			if (itemsetInTransaction(transaction, effectArray)) {
				supportEffect += 1;
			}

			i++;
		}

		return new int[]{supportCause, supportEffect};
	}

	private boolean itemsetInTransaction(Short transaction[],Short itemset[]){
		return ArraysAlgos.includedIn(itemset, transaction) ;
	}

}
