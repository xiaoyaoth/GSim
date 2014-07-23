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
};

struct PredatorAgentData: public BoidAgentData
{
};

struct HahaBoidData : public GAgentData
{
	FLOATn vel;
	FLOATn acc;
	int mass;
};

struct dataUnion
{
	BOID_TYPE bt;
	union {
		PreyAgentData boidAgentData;
		PredatorAgentData predatorAgentData;
		HahaBoidData hahaBoidData;
	};
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
//class PreyAgent;
//class PredatorAgent;
class HahaPreyAgent;
class HahaPredatorAgent;

__global__ void addAgents(BoidModel *bModel);

#define N_POOL 2
__constant__ int N_HAHA_PREY;
int MAX_N_HAHA_PREY;
__constant__ int N_HAHA_PREDATOR;
int MAX_N_HAHA_PREDAOTR;

class BoidModel : public GModel{
public:
	bool poolMod;
	GRandom *random, *randomHost;
	/*
	AgentPool<PreyAgent, PreyAgentData> *pool, *poolHost;
	AgentPool<PredatorAgent, PredatorAgentData> *pool2, *poolHost2;
	AgentPool<PreyAgent, PreyAgentData> *pool3, *poolHost3;
	AgentPool<PredatorAgent, PredatorAgentData> *pool4, *poolHost4;
	*/
	AgentPool<HahaPreyAgent, HahaBoidData> *hahaPreyPool, *hahaPreyPoolHost;
	AgentPool<HahaPredatorAgent, HahaBoidData> *hahaPredatorPool, *hahaPredatorPoolHost;

	cudaEvent_t timerStart, timerStop;
	
	__host__ BoidModel(float range, int numPrey, int numPred)
	{
		poolMod = true;
		/*
		poolHost = new AgentPool<PreyAgent, PreyAgentData>(AGENT_NO / N_POOL, MAX_AGENT_NO / N_POOL, sizeof(dataUnion));
		util::hostAllocCopyToDevice<AgentPool<PreyAgent, PreyAgentData>>(poolHost, &pool);

		
		if (N_POOL > 1) {
			poolHost2 = new AgentPool<PredatorAgent, PredatorAgentData>(AGENT_NO / N_POOL, MAX_AGENT_NO / N_POOL, sizeof(dataUnion));
			util::hostAllocCopyToDevice<AgentPool<PredatorAgent, PredatorAgentData>>(poolHost2, &pool2);
		}

		if (N_POOL > 2) {
			poolHost3 = new AgentPool<PreyAgent, PreyAgentData>(AGENT_NO / N_POOL, MAX_AGENT_NO / N_POOL, sizeof(dataUnion));
			util::hostAllocCopyToDevice<AgentPool<PreyAgent, PreyAgentData>>(poolHost3, &pool3);
		}

		if (N_POOL > 3) {
			poolHost4 = new AgentPool<PredatorAgent, PredatorAgentData>(AGENT_NO / N_POOL, MAX_AGENT_NO / N_POOL, sizeof(dataUnion));
			util::hostAllocCopyToDevice<AgentPool<PredatorAgent, PredatorAgentData>>(poolHost4, &pool4);
		}

		*/
		MAX_N_HAHA_PREDAOTR = numPred;
		MAX_N_HAHA_PREY = numPrey;
		cudaMemcpyToSymbol(N_HAHA_PREDATOR, &numPred, sizeof(int));
		cudaMemcpyToSymbol(N_HAHA_PREY, &numPrey, sizeof(int));

		hahaPreyPoolHost = new AgentPool<HahaPreyAgent, HahaBoidData>(numPrey, MAX_N_HAHA_PREY, sizeof(dataUnion));
		util::hostAllocCopyToDevice<AgentPool<HahaPreyAgent, HahaBoidData>>(hahaPreyPoolHost, &hahaPreyPool);

		hahaPredatorPoolHost = new AgentPool<HahaPredatorAgent, HahaBoidData>(numPred, MAX_N_HAHA_PREDAOTR, sizeof(dataUnion));
		util::hostAllocCopyToDevice<AgentPool<HahaPredatorAgent, HahaBoidData>>(hahaPredatorPoolHost, &hahaPredatorPool);
		
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
		paramHost.neighborhood = range;
		paramHost.jump = 0.7;
		paramHost.maxForce = 6;
		paramHost.maxForcePredator = 10;

		cudaMemcpyToSymbol(params, &paramHost, sizeof(SimParams));
	}

	__host__ void start()
	{
		int AGENT_NO = this->hahaPredatorPoolHost->numElem + this->hahaPreyPoolHost->numElem;
		int gSize = GRID_SIZE(AGENT_NO);
		addAgents<<<gSize, BLOCK_SIZE>>>((BoidModel*)this->model);
#ifdef _WIN32
		GSimVisual::getInstance().setWorld(this->world);
#endif
		cudaEventCreate(&timerStart);
		cudaEventCreate(&timerStop);
		cudaEventRecord(timerStart, 0);
	}

	__host__ void registerPool() 
	{
		/*
		numRegistered += this->poolHost->registerPool(this->worldHost, this->schedulerHost, this->pool, numRegistered);
		if (N_POOL > 1)numRegistered += this->poolHost2->registerPool(this->worldHost, this->schedulerHost, this->pool2, numRegistered);
		if (N_POOL > 2)numRegistered += this->poolHost3->registerPool(this->worldHost, this->schedulerHost, this->pool3, numRegistered);
		if (N_POOL > 3)numRegistered += this->poolHost4->registerPool(this->worldHost, this->schedulerHost, this->pool4, numRegistered);
		*/

		this->hahaPreyPoolHost->registerPool(this->worldHost, this->schedulerHost, this->hahaPreyPool);
		this->hahaPredatorPoolHost->registerPool(this->worldHost, this->schedulerHost, this->hahaPredatorPool);
		cudaMemcpyToSymbol(modelDevParams, &modelHostParams, sizeof(modelConstants));
	}

	__host__ void preStep()
	{
		registerPool();
#ifdef _WIN32
		GSimVisual::getInstance().animate();
#endif
	}

	__host__ void step()
	{
		//int gSize = GRID_SIZE(this->poolHost->numElem);
		int numStepped = 0;

		/*
		numStepped += this->poolHost->stepPoolAgent(this->model, numStepped);
		if (N_POOL > 1)numStepped += this->poolHost2->stepPoolAgent(this->model, numStepped);
		if (N_POOL > 2)numStepped += this->poolHost3->stepPoolAgent(this->model, numStepped);
		if (N_POOL > 3)numStepped += this->poolHost4->stepPoolAgent(this->model, numStepped);
		*/

		numStepped += this->hahaPreyPoolHost->stepPoolAgent(this->model, numStepped);
		numStepped += this->hahaPredatorPoolHost->stepPoolAgent(this->model, numStepped);
	}

	__host__ void stop()
	{
		float time;
		cudaDeviceSynchronize();
		cudaEventRecord(timerStop, 0);
		cudaEventSynchronize(timerStop);
		cudaEventElapsedTime(&time, timerStart, timerStop);
		std::cout<<time<<std::endl;

#ifdef _WIN32
		GSimVisual::getInstance().stop();
#endif
	}
};


//class PreyAgent : public GAgent
//{
//public:
//	BoidModel *model;
//	GRandom *random;
//	AgentPool<PreyAgent, PreyAgentData> *pool;
//
//	__device__ PreyAgent(BoidModel *bModel, AgentPool<PreyAgent, PreyAgentData> *pool, int dataSlot)
//	{
//		this->model = bModel;
//		this->random = bModel->random;
//		this->pool = pool;
//		this->color = colorConfigs.green;
//
//		PreyAgentData *myData = &this->pool->dataArray[dataSlot];
//		PreyAgentData *myDataCopy = &this->pool->dataCopyArray[dataSlot];
//		myData->loc.x = random->uniform() * WIDTH_D;
//		myData->loc.y = random->uniform() * HEIGHT_D;
//		myData->lastd.x = 0;
//		myData->lastd.y = 0;
//		*myDataCopy = *myData;
//
//		this->data = myData;
//		this->dataCopy = myDataCopy;
//	}
//
//	__device__ FLOATn consistency(const GWorld *world, iterInfo &info)
//	{
//		FLOATn res = make_float2(0,0);
//		float ds;
//		FLOATn m;
//		PreyAgentData myData = *(PreyAgentData*)this->data;
//		world->neighborQueryInit(myData.loc, params.neighborhood, info);
//		PreyAgentData otherData;
//		//GAgent *other = world->nextAgent<dataUnion>(info);
//		dataUnion *elem = world->nextAgentDataFromSharedMem<dataUnion>(info);
//		while(elem != NULL){
//			//if (other->agentType != BOID_PREY) {
//			//	other = world->nextAgent<dataUnion>(info);
//			//	continue;
//			//}
//			//elem = (dataUnion*)other->dataInSmem;
//			otherData = elem->boidAgentData;
//			ds = length(myData.loc - otherData.loc);
//			if (ds < params.neighborhood && ds > 0 ) {
//				info.count++;
//				m = otherData.lastd;
//				res = res + m;
//			}
//			//other = world->nextAgent<dataUnion>(info);
//			elem = world->nextAgentDataFromSharedMem<dataUnion>(info);
//		}
//
//		if (info.count > 0){
//			res = res / info.count;
//		}
//
//		return res;
//	}
//
//	__device__ FLOATn cohesion(const GWorld *world, iterInfo &info)
//	{
//		FLOATn res = make_float2(0.0f,0.0f);
//		float ds;
//		FLOATn m;
//		PreyAgentData myData = *(PreyAgentData*)this->data;
//		world->neighborQueryInit(myData.loc, params.neighborhood, info);
//		PreyAgentData otherData;
//		//GAgent *other = world->nextAgent<dataUnion>(info);
//		dataUnion *elem = world->nextAgentDataFromSharedMem<dataUnion>(info);
//		while(elem != NULL){
//			//if (other->agentType != BOID_PREY) {
//			//	other = world->nextAgent<dataUnion>(info);
//			//	continue;
//			//}
//			//elem = (dataUnion*)other->dataInSmem;
//			otherData = elem->boidAgentData;
//			ds = length(myData.loc - otherData.loc);
//			if (ds < params.neighborhood && ds > 0) {
//				info.count++;
//				res = res + myData.loc - otherData.loc;
//			}
//			//other = world->nextAgent<dataUnion>(info);
//			elem = world->nextAgentDataFromSharedMem<dataUnion>(info);
//		}
//
//		if (info.count > 0){
//			res = res / info.count;
//		}
//		res = -res/10;
//		return res;
//	}
//
//	__device__ FLOATn avoidance(const GWorld *world, iterInfo &info)
//	{
//		FLOATn res = make_float2(0,0);
//		FLOATn delta = make_float2(0,0);
//		float ds;
//		PreyAgentData myData = *(PreyAgentData*)this->data;
//		world->neighborQueryInit(myData.loc, params.neighborhood, info);
//		PreyAgentData otherData;
//		//GAgent *other = world->nextAgent<dataUnion>(info);
//		dataUnion *elem = world->nextAgentDataFromSharedMem<dataUnion>(info);
//		while(elem != NULL){
//			//elem = (dataUnion*)other->dataInSmem;
//			otherData = elem->boidAgentData;
//			ds = length(myData.loc - otherData.loc);
//			if (ds < params.neighborhood && ds > 0) {
//				info.count++;
//				delta = myData.loc - otherData.loc;
//				float lensquared = dot(delta, delta);
//				res = res + delta / ( lensquared *lensquared + 1 );
//			}
//			//other = world->nextAgent<dataUnion>(info);
//			elem = world->nextAgentDataFromSharedMem<dataUnion>(info);
//		}
//
//		if (info.count > 0){
//			res = res / info.count;
//		}
//
//		res = res * 400;
//		return res;
//	}
//
//	__device__ FLOATn randomness(){
//		float x = this->random->uniform() * 2 - 1.0;
//		float y = this->random->uniform() * 2 - 1.0;
//		float l = sqrtf(x * x + y * y);
//		FLOATn res;
//		res.x = 0.05 * x / l;
//		res.y = 0.05 * y / l;
//		return res;
//	}
//
//	__device__ void step(GModel *model)
//	{
//		__syncthreads(); //这个barrier可以放到刚进step，但是不能放到getLoc()之后， 不然sync会出错
//		BoidModel *boidModel = (BoidModel*) model;
//		const GWorld *world = boidModel->world;
//		PreyAgentData dataLocal = *(PreyAgentData*)this->data;
//		iterInfo info;
//
//		float dx = 0; 
//		float dy = 0;
//
//		FLOATn cohes = this->cohesion(world, info);
//		FLOATn consi = this->consistency(world, info);
//		FLOATn avoid = this->avoidance(world, info);
//		FLOATn rdnes = this->randomness();
//		FLOATn momen = dataLocal.lastd;
//		dx = 0
//			+ cohes.x * params.cohesion 
//			+ avoid.x * params.avoidance
//			+ consi.x * params.consistency
//			+ rdnes.x * params.randomness
//			+ momen.x * params.momentum
//			;
//		dy = 0
//			+ cohes.y * params.cohesion
//			+ avoid.y * params.avoidance
//			+ consi.y * params.consistency
//			+ rdnes.y * params.randomness
//			+ momen.y * params.momentum
//			;
//
//		float dist = sqrtf(dx*dx + dy*dy);
//		if (dist > 0){
//			dx = dx / dist * params.jump;
//			dy = dy / dist * params.jump;
//		}
//
//
//		PreyAgentData *dummyDataPtr = (PreyAgentData *)this->dataCopy;
//		dummyDataPtr->lastd.x = dx;
//		dummyDataPtr->lastd.y = dy;
//		dummyDataPtr->loc.x = world->stx(dataLocal.loc.x + dx, world->width);
//		dummyDataPtr->loc.y = world->sty(dataLocal.loc.y + dy, world->height);
//	}
//
//	__device__ void setDataInSmem(void *elem)
//	{
//		dataUnion *dataInSmem = (dataUnion*)elem;
//		dataInSmem->bt = BOID_PREY;
//		dataInSmem->boidAgentData = *(PreyAgentData*)this->data;
//	}
//
//	__device__ void die()
//	{
//		this->pool->remove(this->ptrInPool);
//	}
//};
//
//class PredatorAgent : public GAgent
//{
//public:
//	BoidModel *model;
//	GRandom *random;
//	AgentPool<PredatorAgent, PredatorAgentData> *pool;
//
//	__device__ PredatorAgent(BoidModel *bModel, AgentPool<PredatorAgent, PredatorAgentData> *pool, int dataSlot)
//	{
//		this->model = bModel;
//		this->random = bModel->random;
//		this->pool = pool;
//		this->color = colorConfigs.yellow;
//
//		PredatorAgentData *myData = &this->pool->dataArray[dataSlot];
//		PredatorAgentData *myDataCopy = &this->pool->dataCopyArray[dataSlot];
//		myData->loc.x = random->uniform() * WIDTH_D;
//		myData->loc.y = random->uniform() * HEIGHT_D;
//		myData->lastd.x = 0;
//		myData->lastd.y = 0;
//		*myDataCopy = *myData;
//
//		this->data = myData;
//		this->dataCopy = myDataCopy;
//	}
//
//	__device__ FLOATn consistency(const GWorld *world, iterInfo &info)
//	{
//		FLOATn res = make_float2(0,0);
//		float ds;
//		FLOATn m;
//		PredatorAgentData myData = *(PredatorAgentData*)this->data;
//		//world->neighborQueryReset(info);
//		world->neighborQueryInit(myData.loc, params.neighborhood, info);
//		PredatorAgentData otherData;
//		GAgent *other = world->nextAgent<dataUnion>(info);
//		dataUnion *elem;
//		while(other != NULL){
//			//if (other->agentType != BOID_PREDATOR) {
//			//	other = world->nextAgent<dataUnion>(info);
//			//	continue;
//			//}
//			elem = (dataUnion*)other->dataInSmem;
//			otherData = elem->predatorAgentData;
//			ds = length(myData.loc - otherData.loc);
//			if (ds < params.neighborhood && ds > 0) {
//				info.count++;
//				m = otherData.lastd;
//				res = res + m;
//			}
//			other = world->nextAgent<dataUnion>(info);
//		}
//
//		if (info.count > 0){
//			res = res / info.count;
//		}
//
//		return res;
//	}
//
//	__device__ FLOATn cohesion(const GWorld *world, iterInfo &info)
//	{
//		FLOATn res = make_float2(0.0f,0.0f);
//		float ds;
//		FLOATn m;
//		PredatorAgentData myData = *(PredatorAgentData*)this->data;
//		//world->neighborQueryReset(info);
//		world->neighborQueryInit(myData.loc, params.neighborhood, info);
//		PredatorAgentData otherData;
//		GAgent *other = world->nextAgent<dataUnion>(info);
//		dataUnion *elem;
//		while(other != NULL){
//			//if (other->agentType != BOID_PREDATOR) {
//			//	other = world->nextAgent<dataUnion>(info);
//			//	continue;
//			//}
//			elem = (dataUnion*)other->dataInSmem;
//			otherData = elem->predatorAgentData;
//			ds = length(myData.loc - otherData.loc);
//			if (ds < params.neighborhood && ds > 0) {
//				info.count++;
//				res = res + myData.loc - otherData.loc;
//			}
//			other = world->nextAgent<dataUnion>(info);
//		}
//
//		if (info.count > 0){
//			res = res / info.count;
//		}
//		res = -res/10;
//		return res;
//	}
//
//	__device__ FLOATn avoidance(const GWorld *world, iterInfo &info)
//	{
//		FLOATn res = make_float2(0,0);
//		FLOATn delta = make_float2(0,0);
//		float ds;
//		PredatorAgentData myData = *(PredatorAgentData*)this->data;
//		world->neighborQueryInit(myData.loc, params.neighborhood, info);
//		PredatorAgentData otherData;
//		GAgent *other = world->nextAgent<dataUnion>(info);
//		dataUnion *elem;
//		while(other != NULL){
//			elem = (dataUnion*)other->dataInSmem;
//			otherData = elem->predatorAgentData;
//			ds = length(myData.loc - otherData.loc);
//			if (ds < params.neighborhood && ds > 0) {
//				info.count++;
//				delta = myData.loc - otherData.loc;
//				float lensquared = dot(delta, delta);
//				res = res + delta / ( lensquared *lensquared + 1 );
//			}
//			other = world->nextAgent<dataUnion>(info);
//		}
//
//		if (info.count > 0){
//			res = res / info.count;
//		}
//
//		res = res * 400;
//		return res;
//	}
//
//	__device__ FLOATn randomness(){
//		float x = this->random->uniform() * 2 - 1.0;
//		float y = this->random->uniform() * 2 - 1.0;
//		float l = sqrtf(x * x + y * y);
//		FLOATn res;
//		res.x = 0.05 * x / l;
//		res.y = 0.05 * y / l;
//		return res;
//	}
//
//	__device__ void step(GModel *model)
//	{
//		__syncthreads(); //这个barrier可以放到刚进step，但是不能放到getLoc()之后， 不然sync会出错
//		/*
//		BoidModel *boidModel = (BoidModel*) model;
//		const GWorld *world = boidModel->world;
//		PredatorAgentData dataLocal = *(PredatorAgentData*)this->data;
//		iterInfo info;
//
//		float ds = 0, float dsSmallest = world->width;
//		PreyAgent *target = NULL;
//
//		world->neighborQueryInit(dataLocal.loc, params.neighborhood, info, this->pool->numElem);
//
//		GAgent *other = world->nextAgent<dataUnion>(info);
//		PreyAgentData otherData;
//		dataUnion *elem;
//		while(other != NULL){
//		if (other->agentType != BOID_PREY) {
//		other = world->nextAgent<dataUnion>(info);
//		continue;
//		}
//		elem = (dataUnion*)other->dataInSmem;
//		otherData = elem->boidAgentData;
//		ds = length(dataLocal.loc - otherData.loc);
//		if (ds < params.neighborhood && ds > 0) {
//		info.count++;
//		if (ds < dsSmallest) {
//		dsSmallest = ds; 
//		target = (PreyAgent*)other;
//		}
//		}
//		other = world->nextAgent<dataUnion>(info);
//		}
//		if (target && this->random->uniform() < 0.002) {
//		target->die();
//		}
//		*/
//		/*
//		FLOATn avoid = this->avoidance(world, info);
//		FLOATn cohes = this->cohesion(world, info);
//		FLOATn consi = this->consistency(world, info);
//		FLOATn rdnes = this->randomness();
//		FLOATn momen = dataLocal.lastd;
//		float dx = 
//		cohes.x * params.cohesion +
//		avoid.x * params.avoidance +
//		consi.x * params.consistency +
//		rdnes.x * params.randomness +
//		momen.x * params.momentum;
//		float dy = 
//		cohes.y * params.cohesion +
//		avoid.y * params.avoidance +
//		consi.y * params.consistency +
//		rdnes.y * params.randomness +
//		momen.y * params.momentum;
//
//		float dist = sqrtf(dx*dx + dy*dy);
//		if (dist > 0){
//		dx = dx / dist * params.jump;
//		dy = dy / dist * params.jump;
//		}
//
//		PredatorAgentData *dummyDataPtr = (PredatorAgentData *)this->dataCopy;
//		dummyDataPtr->lastd.x = dx;
//		dummyDataPtr->lastd.y = dy;
//		dummyDataPtr->loc.x = world->stx(dataLocal.loc.x + dx, world->width);
//		dummyDataPtr->loc.y = world->sty(dataLocal.loc.y + dy, world->height);
//		*/
//	}
//
//	__device__ void setDataInSmem(void *elem)
//	{
//		dataUnion *dataInSmem = (dataUnion*)elem;
//		dataInSmem->bt = BOID_PREDATOR;
//		dataInSmem->predatorAgentData = *(PredatorAgentData*)this->data;
//	}
//};

class HahaBoid : public GAgent
{
public:
	BoidModel *model;
	GRandom *random;

	__device__ void avoidForce(const GWorld *world, iterInfo &info, int maxForce, HahaBoidData &myData)
	{
		FLOATn locSum = make_float2(0);
		int separation = myData.mass + 20;

		world->neighborQueryInit(myData.loc, separation, info);
		HahaBoidData otherData;
		dataUnion *elem = world->nextAgentDataFromSharedMem<dataUnion>(info);
		float ds = 0;
		while(elem != NULL){
			if (elem->bt != BOID_PREY) {
				elem = world->nextAgentDataFromSharedMem<dataUnion>(info);
				continue;
			}
			otherData = elem->hahaBoidData;//->boidAgentData;
			ds = length(myData.loc - otherData.loc);
			if (ds < separation && ds > 0) {
				info.count++;
				locSum += otherData.loc;
			}
			elem = world->nextAgentDataFromSharedMem<dataUnion>(info);
		}

		if (info.count > 0) {
			locSum /= info.count;
			FLOATn avoidVec = myData.loc - locSum;
			float mag = length(avoidVec);
			if(mag > maxForce * 2.5) avoidVec *= maxForce * 2.5 / mag;
			applyF(avoidVec, myData);
		}
	}

	__device__ void approachForce(const GWorld *world, iterInfo &info, float approachRadius, int maxForce, HahaBoidData &myData)
	{
		FLOATn locSum = make_float2(0);

		world->neighborQueryInit(myData.loc, approachRadius, info);
		HahaBoidData otherData;
		dataUnion *elem = world->nextAgentDataFromSharedMem<dataUnion>(info);
		float ds = 0;

		while(elem != NULL){
			if (elem->bt != BOID_PREY) {
				elem = world->nextAgentDataFromSharedMem<dataUnion>(info);
				continue;
			}
			otherData = elem->hahaBoidData;//->boidAgentData;
			ds = length(myData.loc - otherData.loc);
			if (ds < approachRadius && ds > 0) {
				info.count++;
				locSum += otherData.loc;
			}
			elem = world->nextAgentDataFromSharedMem<dataUnion>(info);
		}

		if (info.count > 0) {
			locSum /= info.count;
			FLOATn approachVec = locSum - myData.loc;
			float mag = length(approachVec);
			if(mag > maxForce) approachVec *= maxForce / mag;
			applyF(approachVec, myData);
		}
	}

	__device__ void alignForce(const GWorld *world, iterInfo &info, int maxForce, HahaBoidData &myData)
	{
		FLOATn velSum = make_float2(0);
		int alignRadius = myData.mass + 100;

		world->neighborQueryInit(myData.loc, alignRadius, info);
		dataUnion *elem = world->nextAgentDataFromSharedMem<dataUnion>(info);
		float ds = 0;

		while(elem != NULL){
			if (elem->bt != BOID_PREY) {
				elem = world->nextAgentDataFromSharedMem<dataUnion>(info);
				continue;
			}
			HahaBoidData otherData = elem->hahaBoidData;//->boidAgentData;
			ds = length(myData.loc - otherData.loc);
			if (ds < alignRadius && ds > 0) {
				info.count++;
				velSum += otherData.vel;
			}
			elem = world->nextAgentDataFromSharedMem<dataUnion>(info);
		}

		if (info.count > 0) {
			velSum /= info.count;
			FLOATn alignVec = velSum;
			float mag = length(alignVec);
			if(mag > maxForce) alignVec *= maxForce / mag;
			applyF(alignVec, myData);
		}
	}

	__device__ void repelForce(FLOATn obstacle, float radius, int maxForce, HahaBoidData &myData)
	{
		FLOATn futPos = myData.loc + myData.vel;
		FLOATn dist = obstacle - futPos;
		float d = length(dist);

		FLOATn repelVec = make_float2(0);

		if (d <=radius){
			repelVec = myData.loc-obstacle;
			repelVec /= length(repelVec);
			if (d != 0) {
				float scale = 1.0 / d;
				float mag = length(repelVec);
				if (mag != 0 ) repelVec *= maxForce * 7 / mag ;
				if (length(repelVec) < 0)
					repelVec.y = 0;
			}
			applyF(repelVec, myData);
		}
	}

	__device__ void repelForceAll(const GWorld *world, iterInfo &info, float radius, int maxForce, HahaBoidData &myData)
	{
		world->neighborQueryInit(myData.loc, radius, info);
		dataUnion *elem = world->nextAgentDataFromSharedMem<dataUnion>(info);
		float ds = 0;

		while(elem != NULL){
			if (elem->bt != BOID_PREDATOR) {
				elem = world->nextAgentDataFromSharedMem<dataUnion>(info);
				continue;
			}
			HahaBoidData otherData = elem->hahaBoidData;//->boidAgentData;
			ds = length(myData.loc - otherData.loc);
			if (ds < radius && ds > 0) {
				this->repelForce(otherData.loc, radius, maxForce, myData);
			}
			elem = world->nextAgentDataFromSharedMem<dataUnion>(info);
		}
	}

	__device__ void applyF(FLOATn &force, HahaBoidData &myData)
	{
		force /= (float)myData.mass;
		myData.acc += force;
	}

	__device__ void correct(float velLimit, float width, float height, HahaBoidData &myData)
	{
		myData.vel += myData.acc;
		myData.loc += myData.vel;
		myData.acc *= 0;
		float mag = length(myData.vel);
		if (mag > velLimit) myData.vel *= velLimit / mag;

		if (myData.loc.x < 0)
			myData.loc.x += width;
		if (myData.loc.x >= width)
			myData.loc.x -= width;
		if (myData.loc.y < 0)
			myData.loc.y += height;
		if (myData.loc.y >= height)
			myData.loc.y -= height;
	}

	__device__ virtual void setDataInSmem(void *elem) = 0;
};

class HahaPreyAgent : public HahaBoid
{
public:
	__device__ HahaPreyAgent(BoidModel *bModel, int dataSlot)
	{
		this->model = bModel;
		this->random = bModel->random;
		this->color = colorConfigs.green;

		HahaBoidData *myData = &bModel->hahaPreyPool->dataArray[dataSlot];
		HahaBoidData *myDataCopy = &bModel->hahaPreyPool->dataCopyArray[dataSlot];

		myData->loc = make_float2(random->uniform() * modelDevParams.WIDTH, random->uniform() * modelDevParams.HEIGHT);
		myData->vel = make_float2(0,0);
		myData->acc = make_float2(0,0);
		myData->mass = 5 * random->uniform() + 5;

		*myDataCopy = *myData;
		this->data = myData;
		this->dataCopy = myDataCopy;
	}
	
	__device__ void step(GModel *model)
	{
		BoidModel *bModel = (BoidModel*)model;
		const GWorld *world = bModel->world;
		HahaBoidData myData = *(HahaBoidData*)this->data;
		iterInfo info;

		float approachRadius = myData.mass + 60;
		float repelRadius = 60;
		int maxForce = params.maxForce;
		this->repelForceAll(world, info, repelRadius, maxForce, myData);
		this->avoidForce(world, info, maxForce, myData);
		this->approachForce(world, info, approachRadius, maxForce, myData);
		this->alignForce(world, info, maxForce, myData);
		
		this->correct(5, world->width, world->height, myData);

		*(HahaBoidData*)this->dataCopy = myData;
	}

	__device__ void setDataInSmem(void *elem)
	{
		dataUnion *dataInSmem = (dataUnion*)elem;
		dataInSmem->bt = BOID_PREY;
		dataInSmem->hahaBoidData = *(HahaBoidData*)this->data;
	}
};

class HahaPredatorAgent : public HahaBoid
{
public:
	__device__ HahaPredatorAgent(BoidModel *bModel, int dataSlot)
	{
		this->model = bModel;
		this->random = bModel->random;
		this->color = colorConfigs.yellow;

		HahaBoidData *myData = &bModel->hahaPredatorPool->dataArray[dataSlot];
		HahaBoidData *myDataCopy = &bModel->hahaPredatorPool->dataCopyArray[dataSlot];

		myData->loc = make_float2(random->uniform() * modelDevParams.WIDTH, random->uniform() * modelDevParams.HEIGHT);
		myData->vel = make_float2(0,0);
		myData->acc = make_float2(0,0);
		myData->mass = 7 * random->uniform() + 8;

		*myDataCopy = *myData;
		this->data = myData;
		this->dataCopy = myDataCopy;
	}

	__device__ void step(GModel *model)
	{
		BoidModel *bModel = (BoidModel*)model;
		const GWorld *world = bModel->world;
		HahaBoidData myData = *(HahaBoidData*)this->data;
		iterInfo info;

		float approachRadius = myData.mass + 260;
		float repelRadius = 30;
		int maxForce = params.maxForcePredator;
		this->repelForceAll(world, info, repelRadius, maxForce, myData);
		this->avoidForce(world, info, maxForce, myData);
		this->approachForce(world, info, approachRadius, maxForce, myData);
		this->alignForce(world, info, maxForce, myData);

		this->correct(6, world->width, world->height, myData);

		*(HahaBoidData*)this->dataCopy = myData;
	}

	__device__ void setDataInSmem(void *elem)
	{
		dataUnion *dataInSmem = (dataUnion*)elem;
		dataInSmem->bt = BOID_PREDATOR;
		dataInSmem->hahaBoidData = *(HahaBoidData*)this->data;
	}
};


//__global__ void addAgents1(BoidModel *model)
//{
//	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
//	int dataSlot = -1;
//	if (idx < AGENT_NO_D / N_POOL) {
//		dataSlot = idx;
//		PreyAgent *ag = new PreyAgent(model, model->pool, dataSlot);
//		model->pool->add(ag, idx);
//	} 
//	else if (N_POOL > 1 && idx < AGENT_NO_D / N_POOL * 2) {
//		dataSlot = idx - AGENT_NO_D / N_POOL ;
//		PredatorAgent *ag = new PredatorAgent(model, model->pool2, dataSlot);
//		model->pool2->add(ag, dataSlot);
//	}
//	else if (N_POOL > 2 && idx < AGENT_NO_D / N_POOL * 3) {
//		dataSlot = idx - AGENT_NO_D / N_POOL * 2 ;
//		PreyAgent *ag = new PreyAgent(model, model->pool3, dataSlot);
//		model->pool3->add(ag, dataSlot);
//	}
//	else if (N_POOL > 3 && idx < AGENT_NO_D / N_POOL * 4) {
//		dataSlot = idx - AGENT_NO_D / N_POOL * 3 ;
//		PredatorAgent *ag = new PredatorAgent(model, model->pool4, dataSlot);
//		model->pool4->add(ag, dataSlot);
//	}
//}

__global__ void addAgents(BoidModel *model)
{
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	int dataSlot = -1;
	if (idx < N_HAHA_PREY) {
		dataSlot = idx;
		HahaPreyAgent *ag = new HahaPreyAgent(model, dataSlot);
		model->hahaPreyPool->add(ag, dataSlot);
	} else if (idx < N_HAHA_PREY + N_HAHA_PREDATOR) {
		dataSlot = idx - N_HAHA_PREY;
		HahaPredatorAgent *ag = new HahaPredatorAgent(model, dataSlot);
		model->hahaPredatorPool->add(ag, dataSlot);
	}
}
#endif