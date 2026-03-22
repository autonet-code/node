const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("GuildRegistry", function () {
    let deployer, creator, member1, member2, member3, reporter;
    let atnToken, guildRegistry;

    const VISUAL_ENCODER = ethers.keccak256(ethers.toUtf8Bytes("visual_encoder"));
    const TEXT_ENCODER = ethers.keccak256(ethers.toUtf8Bytes("text_encoder"));
    const PREDICTOR = ethers.keccak256(ethers.toUtf8Bytes("predictor"));
    const TEXT_DECODER = ethers.keccak256(ethers.toUtf8Bytes("text_decoder"));

    const GUILD_STAKE = ethers.parseEther("500");
    const MEMBER_STAKE = ethers.parseEther("50");
    const INITIAL_SUPPLY = ethers.parseEther("1000000");

    beforeEach(async function () {
        [deployer, creator, member1, member2, member3, reporter] = await ethers.getSigners();

        // Deploy ATN token
        const ATNToken = await ethers.getContractFactory("ATNToken");
        atnToken = await ATNToken.deploy(deployer.address, INITIAL_SUPPLY);

        // Distribute tokens to test accounts
        await atnToken.transfer(creator.address, ethers.parseEther("10000"));
        await atnToken.transfer(member1.address, ethers.parseEther("10000"));
        await atnToken.transfer(member2.address, ethers.parseEther("10000"));
        await atnToken.transfer(member3.address, ethers.parseEther("10000"));

        // Deploy GuildRegistry
        const GuildRegistry = await ethers.getContractFactory("GuildRegistry");
        guildRegistry = await GuildRegistry.deploy(
            await atnToken.getAddress(),
            deployer.address,
            GUILD_STAKE,
            MEMBER_STAKE
        );

        // Set reporter
        await guildRegistry.setReporter(reporter.address, true);
    });

    // =========================================================================
    // Story 8.1: Guild Creation
    // =========================================================================

    describe("Guild Creation", function () {
        it("should create a guild with modules", async function () {
            await atnToken.connect(creator).approve(
                await guildRegistry.getAddress(), GUILD_STAKE
            );

            await expect(
                guildRegistry.connect(creator).createGuild(
                    "Vision Guild",
                    [VISUAL_ENCODER, PREDICTOR]
                )
            )
                .to.emit(guildRegistry, "GuildCreated")
                .withArgs(1, "Vision Guild", creator.address, [VISUAL_ENCODER, PREDICTOR]);

            const guild = await guildRegistry.getGuild(1);
            expect(guild.name).to.equal("Vision Guild");
            expect(guild.moduleIds).to.deep.equal([VISUAL_ENCODER, PREDICTOR]);
            expect(guild.memberCount).to.equal(1);
            expect(guild.aggregator).to.equal(creator.address);
            expect(guild.creator).to.equal(creator.address);
            expect(guild.totalStaked).to.equal(GUILD_STAKE);
            expect(guild.active).to.be.true;
        });

        it("should assign modules to guild", async function () {
            await atnToken.connect(creator).approve(
                await guildRegistry.getAddress(), GUILD_STAKE
            );
            await guildRegistry.connect(creator).createGuild(
                "Vision Guild", [VISUAL_ENCODER]
            );

            expect(await guildRegistry.getModuleGuild(VISUAL_ENCODER)).to.equal(1);
        });

        it("should revert if module already assigned", async function () {
            await atnToken.connect(creator).approve(
                await guildRegistry.getAddress(), GUILD_STAKE * 2n
            );
            await guildRegistry.connect(creator).createGuild(
                "Guild A", [VISUAL_ENCODER]
            );

            await expect(
                guildRegistry.connect(creator).createGuild(
                    "Guild B", [VISUAL_ENCODER]
                )
            ).to.be.revertedWithCustomError(guildRegistry, "ModuleAlreadyAssigned");
        });

        it("should revert with empty name", async function () {
            await atnToken.connect(creator).approve(
                await guildRegistry.getAddress(), GUILD_STAKE
            );
            await expect(
                guildRegistry.connect(creator).createGuild("", [VISUAL_ENCODER])
            ).to.be.revertedWithCustomError(guildRegistry, "InvalidInput");
        });

        it("should revert with no modules", async function () {
            await atnToken.connect(creator).approve(
                await guildRegistry.getAddress(), GUILD_STAKE
            );
            await expect(
                guildRegistry.connect(creator).createGuild("Empty Guild", [])
            ).to.be.revertedWithCustomError(guildRegistry, "NoModules");
        });

        it("should revert if insufficient stake", async function () {
            // No approval
            await expect(
                guildRegistry.connect(creator).createGuild(
                    "Vision Guild", [VISUAL_ENCODER]
                )
            ).to.be.reverted;
        });

        it("should transfer stake from creator", async function () {
            const balBefore = await atnToken.balanceOf(creator.address);
            await atnToken.connect(creator).approve(
                await guildRegistry.getAddress(), GUILD_STAKE
            );
            await guildRegistry.connect(creator).createGuild(
                "Vision Guild", [VISUAL_ENCODER]
            );
            const balAfter = await atnToken.balanceOf(creator.address);
            expect(balBefore - balAfter).to.equal(GUILD_STAKE);
        });

        it("should increment guild IDs", async function () {
            await atnToken.connect(creator).approve(
                await guildRegistry.getAddress(), GUILD_STAKE * 2n
            );
            await guildRegistry.connect(creator).createGuild("G1", [VISUAL_ENCODER]);
            await guildRegistry.connect(creator).createGuild("G2", [TEXT_ENCODER]);

            expect(await guildRegistry.getGuildCount()).to.equal(2);
            const g1 = await guildRegistry.getGuild(1);
            const g2 = await guildRegistry.getGuild(2);
            expect(g1.name).to.equal("G1");
            expect(g2.name).to.equal("G2");
        });
    });

    // =========================================================================
    // Story 8.1: Guild Membership
    // =========================================================================

    describe("Guild Membership", function () {
        beforeEach(async function () {
            await atnToken.connect(creator).approve(
                await guildRegistry.getAddress(), GUILD_STAKE
            );
            await guildRegistry.connect(creator).createGuild(
                "Vision Guild", [VISUAL_ENCODER]
            );
        });

        it("should allow joining a guild", async function () {
            await atnToken.connect(member1).approve(
                await guildRegistry.getAddress(), MEMBER_STAKE
            );
            await expect(
                guildRegistry.connect(member1).joinGuild(1)
            )
                .to.emit(guildRegistry, "MemberJoined")
                .withArgs(1, member1.address, MEMBER_STAKE);

            expect(await guildRegistry.isMember(member1.address, 1)).to.be.true;

            const guild = await guildRegistry.getGuild(1);
            expect(guild.memberCount).to.equal(2);
            expect(guild.totalStaked).to.equal(GUILD_STAKE + MEMBER_STAKE);
        });

        it("should revert if already a member", async function () {
            await atnToken.connect(member1).approve(
                await guildRegistry.getAddress(), MEMBER_STAKE * 2n
            );
            await guildRegistry.connect(member1).joinGuild(1);
            await expect(
                guildRegistry.connect(member1).joinGuild(1)
            ).to.be.revertedWithCustomError(guildRegistry, "AlreadyMember");
        });

        it("should revert joining non-existent guild", async function () {
            await expect(
                guildRegistry.connect(member1).joinGuild(999)
            ).to.be.revertedWithCustomError(guildRegistry, "GuildNotFound");
        });

        it("should allow leaving a guild", async function () {
            await atnToken.connect(member1).approve(
                await guildRegistry.getAddress(), MEMBER_STAKE
            );
            await guildRegistry.connect(member1).joinGuild(1);

            const balBefore = await atnToken.balanceOf(member1.address);
            await expect(
                guildRegistry.connect(member1).leaveGuild(1)
            )
                .to.emit(guildRegistry, "MemberLeft")
                .withArgs(1, member1.address, MEMBER_STAKE);

            const balAfter = await atnToken.balanceOf(member1.address);
            expect(balAfter - balBefore).to.equal(MEMBER_STAKE);

            expect(await guildRegistry.isMember(member1.address, 1)).to.be.false;
            const guild = await guildRegistry.getGuild(1);
            expect(guild.memberCount).to.equal(1);
        });

        it("should revert leaving if not a member", async function () {
            await expect(
                guildRegistry.connect(member1).leaveGuild(1)
            ).to.be.revertedWithCustomError(guildRegistry, "NotMember");
        });

        it("should track guild member list", async function () {
            await atnToken.connect(member1).approve(
                await guildRegistry.getAddress(), MEMBER_STAKE
            );
            await atnToken.connect(member2).approve(
                await guildRegistry.getAddress(), MEMBER_STAKE
            );
            await guildRegistry.connect(member1).joinGuild(1);
            await guildRegistry.connect(member2).joinGuild(1);

            const memberList = await guildRegistry.getGuildMembers(1);
            expect(memberList.length).to.equal(3); // creator + 2 members
            expect(memberList).to.include(creator.address);
            expect(memberList).to.include(member1.address);
            expect(memberList).to.include(member2.address);
        });
    });

    // =========================================================================
    // Story 8.1: Aggregator Election
    // =========================================================================

    describe("Aggregator Election", function () {
        beforeEach(async function () {
            await atnToken.connect(creator).approve(
                await guildRegistry.getAddress(), GUILD_STAKE
            );
            await guildRegistry.connect(creator).createGuild(
                "Vision Guild", [VISUAL_ENCODER]
            );
            await atnToken.connect(member1).approve(
                await guildRegistry.getAddress(), MEMBER_STAKE
            );
            await guildRegistry.connect(member1).joinGuild(1);
        });

        it("should allow creator to set aggregator", async function () {
            await expect(
                guildRegistry.connect(creator).setGuildAggregator(1, member1.address)
            )
                .to.emit(guildRegistry, "AggregatorSet")
                .withArgs(1, member1.address);

            expect(await guildRegistry.isGuildAggregator(member1.address, 1)).to.be.true;
        });

        it("should revert if non-creator sets aggregator", async function () {
            await expect(
                guildRegistry.connect(member1).setGuildAggregator(1, member1.address)
            ).to.be.revertedWithCustomError(guildRegistry, "NotGuildCreator");
        });

        it("should revert setting non-member as aggregator", async function () {
            await expect(
                guildRegistry.connect(creator).setGuildAggregator(1, member2.address)
            ).to.be.revertedWithCustomError(guildRegistry, "NotMember");
        });

        it("should prevent aggregator from leaving if guild has other members", async function () {
            // Creator is default aggregator and guild has member1
            await expect(
                guildRegistry.connect(creator).leaveGuild(1)
            ).to.be.revertedWithCustomError(guildRegistry, "NotAuthorized");
        });

        it("should allow aggregator to leave after transferring role", async function () {
            await guildRegistry.connect(creator).setGuildAggregator(1, member1.address);
            // Now creator can leave since they're no longer aggregator
            await expect(
                guildRegistry.connect(creator).leaveGuild(1)
            ).to.emit(guildRegistry, "MemberLeft");
        });
    });

    // =========================================================================
    // Story 8.4: Guild Reputation
    // =========================================================================

    describe("Guild Reputation", function () {
        beforeEach(async function () {
            await atnToken.connect(creator).approve(
                await guildRegistry.getAddress(), GUILD_STAKE
            );
            await guildRegistry.connect(creator).createGuild(
                "Vision Guild", [VISUAL_ENCODER, PREDICTOR]
            );
        });

        it("should start with base reputation", async function () {
            const rep = await guildRegistry.getReputation(1, VISUAL_ENCODER);
            expect(rep.score).to.equal(5000);
            expect(rep.evaluationCount).to.equal(0);
        });

        it("should record metrics and update reputation", async function () {
            await expect(
                guildRegistry.connect(reporter).recordGuildMetrics(1, VISUAL_ENCODER, 8000)
            )
                .to.emit(guildRegistry, "GuildMetricsRecorded")
                .withArgs(1, VISUAL_ENCODER, 8000, 8000);

            const rep = await guildRegistry.getReputation(1, VISUAL_ENCODER);
            expect(rep.score).to.equal(8000);
            expect(rep.evaluationCount).to.equal(1);
        });

        it("should use EMA for subsequent evaluations", async function () {
            // First: score = 8000
            await guildRegistry.connect(reporter).recordGuildMetrics(1, VISUAL_ENCODER, 8000);

            // Second: EMA = (8000 * 1 + 6000) / 2 = 7000
            await guildRegistry.connect(reporter).recordGuildMetrics(1, VISUAL_ENCODER, 6000);

            const rep = await guildRegistry.getReputation(1, VISUAL_ENCODER);
            expect(rep.score).to.equal(7000);
            expect(rep.evaluationCount).to.equal(2);
        });

        it("should cap EMA effective count at 20", async function () {
            // Submit 25 evaluations at 5000
            for (let i = 0; i < 25; i++) {
                await guildRegistry.connect(reporter).recordGuildMetrics(1, VISUAL_ENCODER, 5000);
            }
            const repBefore = await guildRegistry.getReputation(1, VISUAL_ENCODER);
            expect(repBefore.score).to.equal(5000);

            // Now submit 10000 — with capped EMA (count=20), it should move more
            // new = (5000 * 20 + 10000) / 21 = 110000/21 ≈ 5238
            await guildRegistry.connect(reporter).recordGuildMetrics(1, VISUAL_ENCODER, 10000);
            const repAfter = await guildRegistry.getReputation(1, VISUAL_ENCODER);
            expect(repAfter.score).to.equal(5238); // (5000*20 + 10000) / 21
        });

        it("should revert if non-reporter records metrics", async function () {
            await expect(
                guildRegistry.connect(member1).recordGuildMetrics(1, VISUAL_ENCODER, 8000)
            ).to.be.revertedWithCustomError(guildRegistry, "NotAuthorized");
        });

        it("should revert if score exceeds MAX_SCORE", async function () {
            await expect(
                guildRegistry.connect(reporter).recordGuildMetrics(1, VISUAL_ENCODER, 10001)
            ).to.be.revertedWithCustomError(guildRegistry, "InvalidInput");
        });

        it("should track reputation per module independently", async function () {
            await guildRegistry.connect(reporter).recordGuildMetrics(1, VISUAL_ENCODER, 9000);
            await guildRegistry.connect(reporter).recordGuildMetrics(1, PREDICTOR, 3000);

            const repVis = await guildRegistry.getReputation(1, VISUAL_ENCODER);
            const repPred = await guildRegistry.getReputation(1, PREDICTOR);

            expect(repVis.score).to.equal(9000);
            expect(repPred.score).to.equal(3000);
        });
    });

    // =========================================================================
    // Story 8.4: Guild Ranking & Reward Multiplier
    // =========================================================================

    describe("Guild Ranking & Multiplier", function () {
        beforeEach(async function () {
            await atnToken.connect(creator).approve(
                await guildRegistry.getAddress(), GUILD_STAKE
            );
            await guildRegistry.connect(creator).createGuild(
                "Vision Guild", [VISUAL_ENCODER]
            );
        });

        it("should return ranking for a module", async function () {
            await guildRegistry.connect(reporter).recordGuildMetrics(1, VISUAL_ENCODER, 7500);

            const [ids, scores] = await guildRegistry.getGuildRanking(VISUAL_ENCODER);
            expect(ids.length).to.equal(1);
            expect(ids[0]).to.equal(1);
            expect(scores[0]).to.equal(7500);
        });

        it("should return empty ranking for unassigned module", async function () {
            const [ids, scores] = await guildRegistry.getGuildRanking(TEXT_DECODER);
            expect(ids.length).to.equal(0);
            expect(scores.length).to.equal(0);
        });

        it("should compute reward multiplier from reputation", async function () {
            // Base reputation is 5000 → multiplier = 5000 + 5000 = 10000 (1x)
            const baseMult = await guildRegistry.getGuildRewardMultiplier(1, VISUAL_ENCODER);
            expect(baseMult).to.equal(10000);

            // High reputation: 9000 → 5000 + 9000 = 14000 (1.4x)
            await guildRegistry.connect(reporter).recordGuildMetrics(1, VISUAL_ENCODER, 9000);
            const highMult = await guildRegistry.getGuildRewardMultiplier(1, VISUAL_ENCODER);
            expect(highMult).to.equal(14000);
        });

        it("should give minimum multiplier for zero reputation", async function () {
            await guildRegistry.connect(reporter).recordGuildMetrics(1, VISUAL_ENCODER, 0);
            const mult = await guildRegistry.getGuildRewardMultiplier(1, VISUAL_ENCODER);
            expect(mult).to.equal(5000); // 0.5x
        });

        it("should give maximum multiplier for perfect reputation", async function () {
            await guildRegistry.connect(reporter).recordGuildMetrics(1, VISUAL_ENCODER, 10000);
            const mult = await guildRegistry.getGuildRewardMultiplier(1, VISUAL_ENCODER);
            expect(mult).to.equal(15000); // 1.5x
        });
    });

    // =========================================================================
    // Admin Functions
    // =========================================================================

    describe("Admin", function () {
        it("should set reporter", async function () {
            await expect(
                guildRegistry.setReporter(member1.address, true)
            ).to.emit(guildRegistry, "ReporterSet")
                .withArgs(member1.address, true);

            expect(await guildRegistry.reporters(member1.address)).to.be.true;
        });

        it("should update min stakes", async function () {
            await guildRegistry.setMinGuildStake(ethers.parseEther("1000"));
            expect(await guildRegistry.minGuildStake()).to.equal(ethers.parseEther("1000"));

            await guildRegistry.setMinMemberStake(ethers.parseEther("100"));
            expect(await guildRegistry.minMemberStake()).to.equal(ethers.parseEther("100"));
        });

        it("should transfer admin", async function () {
            await guildRegistry.transferAdmin(creator.address);
            expect(await guildRegistry.admin()).to.equal(creator.address);
        });

        it("should revert non-admin actions", async function () {
            await expect(
                guildRegistry.connect(member1).setReporter(member1.address, true)
            ).to.be.revertedWithCustomError(guildRegistry, "NotAuthorized");

            await expect(
                guildRegistry.connect(member1).transferAdmin(member1.address)
            ).to.be.revertedWithCustomError(guildRegistry, "NotAuthorized");
        });
    });
});
