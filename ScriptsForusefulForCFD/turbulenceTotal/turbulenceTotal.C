/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2013-2016 OpenFOAM Foundation
    Copyright (C) 2015-2022 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/
#include "turbulenceTotal.H"
#include "turbulentTransportModel.H"
#include "addToRunTimeSelectionTable.H"
// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace functionObjects
{
    defineTypeNameAndDebug(turbulenceTotal, 0);
    addToRunTimeSelectionTable(functionObject, turbulenceTotal, dictionary);
}
}


const Foam::Enum
<
    Foam::functionObjects::turbulenceTotal::incompressibleField
>
Foam::functionObjects::turbulenceTotal::incompressibleFieldNames_
({
    { incompressibleField::ifK, "k" }
   // { incompressibleField::ifEpsilon, "epsilon" },
   // { incompressibleField::ifOmega, "omega" },

});


const Foam::word Foam::functionObjects::turbulenceTotal::modelName_
(
    Foam::turbulenceModel::propertiesName
);


// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

void Foam::functionObjects::turbulenceTotal::initialise()
{
    for (const word& f : fieldSet_)
    {
        const word localName(IOobject::scopedName(prefix_, f));

        if (obr_.found(localName))
        {
            WarningInFunction
                << "Cannot store turbulence field " << localName
                << " since an object with that name already exists"
                << nl << endl;

            fieldSet_.unset(f);
        }
    }

    initialised_ = true;
}



// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::functionObjects::turbulenceTotal::turbulenceTotal
(
    const word& name,
    const Time& runTime,
    const dictionary& dict
)
:
    fvMeshFunctionObject(name, runTime, dict),
    initialised_(false),
    prefix_(dict.getOrDefault<word>("prefix", "turbulenceProperties")),
    fieldSet_()
{
    read(dict);
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

bool Foam::functionObjects::turbulenceTotal::read(const dictionary& dict)
{
    Foam::Info << "FO created by STFS-- Sritej under Jannis Reusch" << Foam::nl;
    if (fvMeshFunctionObject::read(dict))
    {
        dict.readIfPresent("prefix", prefix_);

        if (dict.found("field"))
        {
            fieldSet_.insert(dict.get<word>("field"));
        }
        else
        {
            fieldSet_.insert(dict.get<wordList>("fields"));
        }

        Foam::Info<< type() << " " << name() << ": ";
        if (fieldSet_.size())
        {
            Foam::Info<< "storing fields:" << Foam::nl;
            for (const word& f : fieldSet_)
            {
                Foam::Info<< "    " << IOobject::scopedName(prefix_, f) << nl;
            }
            Foam::Info<< endl;
        }
        else
        {
            Foam::Info<< "no fields requested to be stored" << Foam::nl << endl;
        }

        initialised_ = false;

        return true;
    }

    return false;
}


bool Foam::functionObjects::turbulenceTotal::execute()
{
    
    if (!initialised_)
    {
        initialise();
    }
        const auto& model = obr_.lookupObject<incompressible::turbulenceModel>(modelName_);

        for (const word& f : fieldSet_)
        {
            if (f == "k")
            {
                volScalarField kResolved
                (
                    IOobject("kResolved", mesh_.time().timeName(), mesh_,
                             IOobject::NO_READ, IOobject::AUTO_WRITE),
                    mesh_,
                    dimensionedScalar("zero", dimVelocity * dimVelocity, 0)
                );
                computeKResolved(kResolved);
                processField<scalar>("kResolved", tmp<volScalarField>(kResolved));

                volScalarField totalK
                (
                    IOobject("totalK", mesh_.time().timeName(), mesh_,
                             IOobject::NO_READ, IOobject::AUTO_WRITE),
                    mesh_,
                    dimensionedScalar("zero", dimVelocity*dimVelocity, 0)
                );
                computeTotalK(kResolved, model.k(), totalK);
                processField<scalar>("totalK", tmp<volScalarField>(totalK));
            }

        }

        Info << "turbulenceTotal executed at time = " << mesh_.time().timeName() << nl;
        return true;
        




   }

    


/*bool Foam::functionObjects::turbulenceTotal::execute()
{
    if (!initialised_)
    {
        initialise();
    }

    const bool comp = compressible();

    if (comp)
    {
        const auto& model =
            obr_.lookupObject<compressible::turbulenceModel>(modelName_);

        for (const word& f : fieldSet_)
        {
            switch (compressibleFieldNames_[f])
            {
                case cfK:
                {
                    processField<scalar>(f, model.k());
                    break;
                }
                case cfEpsilon:
                {
                    processField<scalar>(f, model.epsilon());
                    break;
                }
                case cfOmega:
                {
                    processField<scalar>(f, model.omega());
                    break;
                }
                case cfR:
                {
                    processField<symmTensor>(f, model.R());
                    break;
                }

                case cfL:
                {
                    processField<scalar>(f, L(model));
                    break;
                }
                case cfI:
                {
                    processField<scalar>(f, I(model));
                    break;
                }

                default:
                {
                    FatalErrorInFunction
                        << "Invalid field selection" << abort(FatalError);
                }
            }
        }
    }
    else
    {
        const auto& model =
            obr_.lookupObject<incompressible::turbulenceModel>(modelName_);

        for (const word& f : fieldSet_)
        {
            switch (incompressibleFieldNames_[f])
            {
                case ifK:
                {
                    processField<scalar>(f, model.k());
                    break;
                }
                case ifEpsilon:
                {
                    processField<scalar>(f, model.epsilon());
                    break;
                }
                case ifOmega:
                {
                    processField<scalar>(f, model.omega());
                    break;
                }


                case ifR:
                {
                    processField<symmTensor>(f, model.R());
                    break;
                }

                case ifL:
                {
                    processField<scalar>(f, L(model));
                    break;
                }
                case ifI:
                {
                    processField<scalar>(f, I(model));
                    break;
                }

                default:
                {
                    FatalErrorInFunction
                        << "Invalid field selection" << abort(FatalError);
                }
            }
        }
    }

    return true;
}*/


bool Foam::functionObjects::turbulenceTotal::write()
{
    for (const word& f : fieldSet_)
    {
        const word localName(IOobject::scopedName(prefix_, f));

        writeObject(localName);
    }
    Info<< endl;

    return true;
}

 // ************************************************************************* //


