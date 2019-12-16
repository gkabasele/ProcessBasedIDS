package algorithms;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.*;

public class InputVerifier {
    Map<ItemSet, Integer> mapPosition;

    public InputVerifier() {
        mapPosition = new LinkedHashMap<>();
    }

    public String toString(){
        StringBuilder str = new StringBuilder();
        for (Map.Entry entry : mapPosition.entrySet()){
            str.append(entry.getKey() + ":" + entry.getValue() + "\n");
        }
        return str.toString();
    }

    public void verifyInput(String filename) throws IOException {
        FileInputStream inputStream = null;
        Scanner sc = null;

        String line;
        String[] it;
        String[] values;
        String[] itemsID;
        short support;
        Set<Short> items;
        ItemSet currentItemSet = null;
        ItemSet itemSet = null;
        int lineNumber = 0;
        try {
            inputStream = new FileInputStream(filename);
            sc = new Scanner(inputStream, "UTF-8");
            while (sc.hasNextLine()) {
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
                itemSet = new ItemSet(items, support);
                if (currentItemSet != null){
                    if(!itemSet.isSupersetOf(currentItemSet)){
                        if (items.size() == 1){
                            mapPosition.put(itemSet, lineNumber);
                            currentItemSet = itemSet;
                        } else {
                            System.out.println("Non superset of " + currentItemSet + "with size greater than one found: " + itemSet);
                            System.out.println("Line :" + lineNumber);
                            return;
                        }
                    }
                } else if (items.size() == 1) {
                    currentItemSet = itemSet;
                    mapPosition.put(currentItemSet, lineNumber);
                }
                lineNumber++;
            }

            if (sc.ioException() != null){
                throw sc.ioException();
            }
        } finally {
            if (inputStream != null){
                inputStream.close();
            }
            if (sc != null){
                sc.close();
            }
        }
    }
}
