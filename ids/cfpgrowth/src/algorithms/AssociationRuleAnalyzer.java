package algorithms;

import java.io.*;
import java.util.*;

public class AssociationRuleAnalyzer {

    private Map<Short, List<AssociationRule>> map;
    private Map<Short, List<AssociationRule>> mapFilter;

    private List<AssociationRule> bigEffectRules;

    public AssociationRuleAnalyzer(){
        map = new HashMap<>();
        mapFilter = new HashMap<>();
        bigEffectRules = new ArrayList<>();
    }

    public void fillMap(String filename) throws IOException {
        String line = null;
        try (FileInputStream inputStream = new FileInputStream(filename); Scanner sc = new Scanner(inputStream, "UTF-8")) {
           while (sc.hasNextLine()){
               line = sc.nextLine();
               AssociationRule rule = new AssociationRule(line);
               if (rule.getEffect().size() == 1){
                  Short item = rule.getEffect().get(0);
                  if (map.containsKey(item)){
                      map.get(item).add(rule);
                  } else {
                      List<AssociationRule> list = new ArrayList<>();
                      list.add(rule);
                      map.put(item, list);
                  }
               } else {
                   bigEffectRules.add(rule);
                   System.out.println("Found Rules with a more than one effect");
                   System.out.println(rule);
               }

           }
        }
    }

    public void createFilterMap(){
       for (Short item : map.keySet()){
           createFilterMap(item, map.get(item));
       }
    }

    public void createFilterMap(Short item, List<AssociationRule> rules){
        List<AssociationRule> filteredList = new ArrayList<>();
        int i = 0;
        for(AssociationRule rule: rules){
            boolean hasSubset = false;
            int j = 0;
            for (AssociationRule other: rules){
                if (i != j && rule.causeSuperserOf(other)){
                    hasSubset = true;
                    break;
                }
                j++;
            }
            if (!hasSubset){
                filteredList.add(rule);
                System.out.println("Adding rule in filtering map: " + rule);
            }
            i++;
        }

        mapFilter.put(item, filteredList);
    }

    public void exportFilteredMap(String filename) throws IOException{
        File wfile = new File(filename);

        if (!wfile.exists()){
            wfile.createNewFile();
        }

        BufferedWriter bw = new BufferedWriter((new FileWriter(filename)));
        for(Short item : mapFilter.keySet()){
            for(AssociationRule rule : mapFilter.get(item)){
                bw.write(rule.toString());
                bw.write("\n");
            }
        }

        System.out.println("Exporting multi-effect rules");

        for (AssociationRule rule : bigEffectRules){
            bw.write(rule.toString());
            bw.write("\n");
        }
        bw.close();
    }

}
