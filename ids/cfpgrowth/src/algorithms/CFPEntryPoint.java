package algorithms;

import py4j.GatewayServer;
public class CFPEntryPoint {

    private AlgoCFPGrowth cfp;
    private AssociationRuleMining miner;

    public CFPEntryPoint(){
        cfp = new AlgoCFPGrowth();
        miner = new AssociationRuleMining();
    }

    public AlgoCFPGrowth getCFP(){
        return cfp;
    }

    public AssociationRuleMining getMiner(){
        return miner;
    }

    public  static void main(String[] args){
        GatewayServer gatewayServer = new GatewayServer(new CFPEntryPoint());
        gatewayServer.start();
        System.out.println("Gateway Server Started");
    }
}
