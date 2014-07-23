#ifndef BOID_CUH
#define BOID_CUH

#include "gsimcore.cuh"
#ifdef _WIN32
#include "gsimvisual.cuh"
#endif

enum BOID_TYPE {BOID_PREY, BOID_PREDATOR};

struct BoidAgentData : public GAgentData
{
	FLOATn lastd;
};

struct PreyAgentData : public BoidAgentData
{
	__device__ void putDataInSmem(GAgent *ag);
};

struct SimParams
{
	float cohesion;
	float avoidance;
	float randomness;
	float consistency;
	float momentum;
	float deadFlockerProbability;
	float neighborhood;
	float jump;

	float maxForce;
	float maxForcePredator;
};

__constant__ SimParams params;

class BoidModel;
class PreyAgent;

__global__ void addAgents(BoidModel *bModel);

__constant__ int N_HAHA_PREY;
int MAX_N_HAHA_PREY;

class BoidModel : public GModel{
public:
	GRandom *random, *randomHost;

	AgentPool<PreyAgent, PreyAgentData> *pool, *poolHost;
	
	cudaEvent_t timerStart, timerStop;
	
	__host__ BoidModel(int numPrey)
	{
		MAX_N_HAHA_PREY = numPrey*2;
		cudaMemcpyToSymbol(N_HAHA_PREY, &numPrey, sizeof(int));

		poolHost = new AgentPool<PreyAgent, PreyAgentData>(numPrey, MAX_N_HAHA_PREY, sizeof(PreyAgentData));
		util::hostAllocCopyToDevice<AgentPool<PreyAgent, PreyAgentData>>(poolHost, &pool);

		worldHost = new GWorld(modelHostParams.WIDTH, modelHostParams.HEIGHT);
		util::hostAllocCopyToDevice<GWorld>(worldHost, &world);

		randomHost = new GRandom(modelHostParams.MAX_AGENT_NO);
		util::hostAllocCopyToDevice<GRandom>(randomHost, &random);

		util::hostAllocCopyToDevice<BoidModel>(this, (BoidModel**)&this->model);

		SimParams paramHost;
		paramHost.cohesion = 1.0;
		paramHost.avoidance = 1.0;
		paramHost.randomness = 1.0;
		paramHost.consistency = 10.0;
		paramHost.momentum = 1.0;
		paramHost.deadFlockerProbability = 0.1;
		paramHost.neighborhood = 0;
		paramHost.jump = 0.7;
		paramHost.maxForce = 6;
		paramHost.maxForcePredator = 10;

		cudaMemcpyToSymbol(params, &paramHost, sizeof(SimParams));
	}

	__host__ void start()
	{
		int AGENT_NO = this->poolHost->numElem;
		int gSize = GRID_SIZE(AGENT_NO);
		addAgents<<<gSize, BLOCK_SIZE>>>((BoidModel*)this->model);
		GSimVisual::getInstance().setWorld(this->world);
		cudaEventCreate(&timerStart);
		cudaEventCreate(&timerStop);
		cudaEventRecord(timerStart, 0);
	}

	__host__ void registerPool() 
	{
		this->poolHost->registerPool(this->worldHost, this->schedulerHost, this->pool);
		cudaMemcpyToSymbol(modelDevParams, &modelHostParams, sizeof(modelConstants));
	}

	__host__ void preStep()
	{
		registerPool();
		GSimVisual::getInstance().animate();
	}

	__host__ void step()
	{
		int gSize = GRID_SIZE(this->poolHost->numElem);
		int numStepped = 0;

		numStepped += this->poolHost->stepPoolAgent(this->model, numStepped);
	}

	__host__ void stop()
	{
		float time;
		cudaDeviceSynchronize();
		cudaEventRecord(timerStop, 0);
		cudaEventSynchronize(timerStop);
		cudaEventElapsedTime(&time, timerStart, timerStop);
		std::cout<<time<<std::endl;

		GSimVisual::getInstance().stop();
	}
};

class PreyAgent : public GAgent
{
public:
	BoidModel *model;
	GRandom *random;
	AgentPool<PreyAgent, PreyAgentData> *pool;

	__device__ void init(BoidModel *bModel, int dataSlot)
	{
		this->model = bModel;
		this->random = bModel->random;
		this->pool = bModel->pool;
		this->color = colorConfigs.green;

		PreyAgentData *myData = &this->pool->dataArray[dataSlot];
		PreyAgentData *myDataCopy = &this->pool->dataCopyArray[dataSlot];
		myData->loc.x = random->uniform() * modelDevParams.WIDTH;
		myData->loc.y = random->uniform() * modelDevParams.HEIGHT;
		myData->lastd.x = 0;
		myData->lastd.y = 0;
		*myDataCopy = *myData;

		this->data = myData;
		this->dataCopy = myDataCopy;
	}

	__device__ void step(GModel *model)
	{
		BoidModel *boidModel = (BoidModel*) model;
		this->pool->remove(this->ptrInPool);
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		int agentSlot = this->pool->numElem + idx;
		int dataSlot = this->pool->dataSlot(agentSlot);
		PreyAgent *ag = &this->pool->agentArray[dataSlot];
		ag->init(boidModel, dataSlot);
		this->pool->add(ag, agentSlot);
	}

};

__global__ void addAgents(BoidModel *model)
{
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	int dataSlot = -1;
	if (idx < N_HAHA_PREY) {
		dataSlot = idx;
		PreyAgent *ag = &model->pool->agentArray[dataSlot];
		ag->init(model, dataSlot);
		model->pool->add(ag, dataSlot);
	}
}

__device__ void PreyAgentData::putDataInSmem(GAgent *ag){
	*this = *(PreyAgentData*)ag->data;
}

#endif